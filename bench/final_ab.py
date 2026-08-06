"""Итоговый прогон: четыре руки, длинное обучение, мощность под значимость.

Скрин на 500 шагах показал направление, но на нём члены CFM не могут показать
себя: согласовывать нечего, прямая задача ещё не выучена. Здесь 2500 шагов и
120 ячеек замера на клетку (5 сидов x 24 промпта), что даёт MDE около 0.05
токена при наблюдённой спаренной sd.

Руки:
  orthrus         маскирующий бейзлайн, его же multi-step denoising на n>1
  fd_base         verify_kl + веса позиций, без CFM, без selfcorrect
  fd_ms           то же + selfcorrect (расписание рестартов)
  fd_cfm_ms       то же + selfcorrect + члены CFM (endpoint/EC/TD)

Проверяемое:
  H1  fd_* - orthrus по A_1: паритет держится (нижняя граница CI выше -0.1)
  H2  TPF(2) и TPF(4) у лучшей руки выше, чем у orthrus на тех же n
  H3  члены CFM платят на ДЛИННОМ расписании: (A_4 - A_1) у fd_cfm_ms выше,
      чем у fd_ms. Это прямой тест гипотезы «CFM делает семейство траекторией».

  STEPS=2500 uv run python .work/final_ab.py
"""
import os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = int(os.environ.get("STEPS", 2500))
ROOT = "/tmp/fd-final"

COMMON = [
    "model=smollm2_135m", "train.block_size=8", "train.anchors_per_sequence=1",
    "train.lr=3e-4", "train.val_decode_prompts=0", "train.early_stop_patience=0",
    "train.monitor=val/loss", "train.monitor_mode=min",
    f"train.checkpoint_every_n_steps={STEPS}",
    "data.batch_size=2", "data.max_length=256", "data.val_size=8",
    # Стрим буферизует shuffle_buffer примеров ДО первого батча, и это
    # доминировало над самим обучением: пять рук платили этот фиксированный
    # налог по разу каждая, а процесс всё это время стоял на 0% CPU и выглядел
    # зависшим. Для прогона на 2000 шагов качество перемешивания столько не
    # стоит.
    "data.shuffle_buffer=64",
    f"trainer.max_steps={STEPS}", f"trainer.val_check_interval={STEPS}",
    "trainer.precision=32", "trainer.log_every_n_steps=100",
]
PW = ["train.teacher_chain_tail_weight=0.3",
      "train.position_weights=[2.32,1.75,1.41,0.87,0.39,0.19,0.07]"]
FD = ["train.variant=flowdraft_block_wise", "train.prior_type=discunif",
      "train.verify_kl_weight=1.0", *PW]
NO_CFM = ["train.endpoint_weight=0.0", "train.lambda=0.0",
          "train.terminal_time_fraction=1.0"]
WITH_CFM = ["train.endpoint_weight=0.5", "train.lambda=0.25",
            "train.terminal_time_fraction=0.25", "train.lambda_ramp_steps=500"]
SELF = ["train.selfcorrect_kl_weight=1.0", "train.selfcorrect_rounds=2",
        "train.selfcorrect_s_min=0.0"]

# Квадрат, разделяющий вклад: {маскирующее состояние, симплексное} x
# {одношаговое обучение, многошаговое}. Контраст по строке даёт вклад ОБУЧЕНИЯ
# под многошаговость, по столбцу — вклад НЕПРЕРЫВНОГО состояния. Если
# orthrus_ms догоняет fd_ms, симплекс не несущий, и это надо знать.
ARMS = {
    "orthrus": ["train.variant=orthrus"],
    "orthrus_ms": ["train.variant=orthrus", *SELF[:1], "train.selfcorrect_tail_weight=0.5"],
    "fd_base": [*FD, *NO_CFM],
    "fd_ms": [*FD, *NO_CFM, *SELF],
    "fd_cfm_ms": [*FD, *WITH_CFM, *SELF],
}

def trained_steps(path):
    """Сколько шагов реально сделано. 0 или отсутствие файла = провал."""
    if not os.path.exists(path):
        return 0
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("global_step", 0))


for name, extra in ARMS.items():
    out = f"{ROOT}/{name}"
    if trained_steps(f"{out}/last.ckpt"):
        print(f"[{name}] чекпоинт есть ({trained_steps(f'{out}/last.ckpt')} шагов)", flush=True)
        continue
    print(f"[{name}] обучение {STEPS} шагов...", flush=True)
    # Процесс ЗАВЕРШАЕТ обучение, пишет чекпоинт и после этого намертво виснет
    # в деструкторе пула потоков pyarrow — он никогда не возвращается. Поэтому
    # готовность определяется по global_step в чекпоинте, а не по выходу
    # процесса: как только шаги набраны, процесс снимается. Ожидание его выхода
    # стоило бы двух часов на каждой уже готовой руке.
    log = f"{out}.trainlog"
    deadline = time.time() + 25 * 60 + STEPS * 2
    with open(log, "w") as sink:
        proc = subprocess.Popen([sys.executable, "src/train.py", *COMMON, *extra,
                                 f"output_dir={out}"], stdout=sink,
                                stderr=subprocess.STDOUT, text=True)
    while True:
        if trained_steps(f"{out}/last.ckpt") >= STEPS:
            proc.kill(); proc.wait(timeout=30); break
        if proc.poll() is not None:      # умер сам, не добрав шагов
            break
        if time.time() > deadline:
            proc.kill(); print(f"[{name}] таймаут", flush=True); break
        time.sleep(15)
    r = type("R", (), {"stdout": "", "stderr": open(log).read()[-3000:]})()
    steps = trained_steps(f"{out}/last.ckpt")
    if not steps:
        # Существование last.ckpt НЕ доказывает обучение: Lightning пишет его и
        # при разрушении после исключения. Ровно так бейзлайн Orthrus простоял
        # на global_step=0, а харнесс считал руку обученной и шёл дальше.
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
        raise SystemExit(f"[{name}] ПРОВАЛ: обучение не продвинулось (шагов: {steps})")
    print(f"[{name}] обучена, {steps} шагов", flush=True)
print("TRAINED_ALL")
