"""Ночной прогон: длинный горизонт и отбор чекпоинта по целевой метрике.

Кривые обучения на 6000 шагов показали, что сравнение там несправедливо в обе
стороны. Одно-компонентный `orthrus` последнюю треть идёт ВВЕРХ по лоссу
(+0.270) -- он уже переобучается, и его превосходство снято в точке, где его
пора было останавливать. Многокомпонентные ещё падают, и тем быстрее, чем
больше членов: -0.073 при двух, -0.733 с самокоррекцией, -1.909 при четырёх;
их приёмка на обучении в тот же период растёт.

Отсюда два изменения против дневного набора:
  * горизонт 12000 с расписанием, заданным ОДИН раз, а не возобновлением:
    возобновление пересобирает косинус и подбрасывает LR обратно к пику;
  * валидационный декод включён, отбор по `val/tpf`, три лучшие точки на конфигурацию.
    На 6000 бралась последняя точка каждой конфигурации -- при переобучающемся
    бейзлайне это искажает сравнение в его пользу по построению.

Один сид: шесть конфигураций на 12000 дважды в ночь не помещаются. Второй -- следом.

  SEEDS=42 STEPS=12000 ROOT=/tmp/fd-night uv run python bench/night.py
"""
import os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEPS = int(os.environ.get("STEPS", 6000))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "42,43").split(",")]
ROOT = os.environ.get("ROOT", "/tmp/fd-full")

COMMON = [
    "model=smollm2_135m", "train.block_size=8", "train.anchors_per_sequence=1",
    "train.lr=3e-4", "train.val_decode_prompts=8", "train.early_stop_patience=0",
    "train.monitor=val/tpf", "train.monitor_mode=max",
    "data.batch_size=2", "data.max_length=256", "data.val_size=8",
    "data.shuffle_buffer=64",
    f"trainer.max_steps={STEPS}", "trainer.val_check_interval=2000",
    "trainer.precision=32", "trainer.log_every_n_steps=200",
    "train.checkpoint_every_n_steps=2000", "train.checkpoint_save_top_k=3",
    "train.checkpoint_name='snap-{step:07d}'",
]
# Веса позиций теперь у ОБЕИХ ветвей: dE/da_j — свойство метрики, не ветви.
PW = ["train.teacher_chain_tail_weight=0.3",
      "train.position_weights=[2.32,1.75,1.41,0.87,0.39,0.19,0.07]"]
FLOW = ["train.variant=flowdraft_block_wise", "train.prior_type=discunif",
        "train.verify_kl_weight=1.0", *PW]
NO_CFM = ["train.endpoint_weight=0.0", "train.lambda=0.0",
          "train.terminal_time_fraction=1.0"]
CFM = ["train.endpoint_weight=0.5", "train.lambda=0.25",
       "train.terminal_time_fraction=0.25", "train.lambda_ramp_steps=500"]
SC = ["train.selfcorrect_kl_weight=1.0", "train.selfcorrect_rounds=2",
      "train.selfcorrect_s_min=0.0", "train.selfcorrect_tail_weight=0.5"]

ARMS = {
    "orthrus":     ["train.variant=orthrus", *PW],
    "flow_sc":     [*FLOW, *NO_CFM, *SC],
    "flow_sc_cfm": [*FLOW, *CFM, *SC],
    "orthrus_sc":  ["train.variant=orthrus", *PW, "train.selfcorrect_kl_weight=1.0"],
    "flow":        [*FLOW, *NO_CFM],
    # Контроль на мою собственную правку: старый профиль весов, где полный вес
    # приходится на принятый префикс, а не на позицию разрыва.
    "flow_sc_accw": [*FLOW, *NO_CFM, *SC, "train.selfcorrect_tail_weight=1.0"],
    # Контроль: маскирующая самокоррекция ДО правки утечки, но с весами позиций.
    # Раскладывает вчерашнее преимущество склейки на «у бейзлайна не было весов»
    # и «в члене была утечка».
    "orthrus_sc_leaky": ["train.variant=orthrus", *PW,
                         "train.selfcorrect_kl_weight=1.0",
                         "+train.selfcorrect_leaky=true"],
}


def steps_of(path):
    if not os.path.exists(path):
        return 0
    import torch
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False)
                   .get("global_step", 0))
    except Exception:
        # Чекпоинт могут писать прямо сейчас — это не провал, это гонка.
        return 0


for seed in SEEDS:
    for name, extra in ARMS.items():
        out = f"{ROOT}/s{seed}/{name}"
        if steps_of(f"{out}/last.ckpt") >= STEPS:
            print(f"[s{seed}/{name}] готово", flush=True)
            continue
        print(f"[s{seed}/{name}] обучение {STEPS} шагов...", flush=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        log = f"{out}.trainlog"
        with open(log, "w") as sink:
            proc = subprocess.Popen(
                [sys.executable, "src/train.py", *COMMON, *extra,
                 f"seed={seed}", f"output_dir={out}"],
                stdout=sink, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 40 * 60 + STEPS * 3
        while True:
            # Готовность читается из чекпоинта: процесс доучивается, пишет его и
            # только потом намертво виснет в деструкторе pyarrow.
            if steps_of(f"{out}/last.ckpt") >= STEPS:
                proc.kill(); proc.wait(timeout=30); break
            if proc.poll() is not None:
                break
            if time.time() > deadline:
                proc.kill(); print(f"[s{seed}/{name}] таймаут", flush=True); break
            time.sleep(20)
        got = steps_of(f"{out}/last.ckpt")
        if got < STEPS:
            print(open(log).read()[-2000:])
            raise SystemExit(f"[s{seed}/{name}] ПРОВАЛ: {got} из {STEPS}")
        print(f"[s{seed}/{name}] готово, {got} шагов", flush=True)
print("FULL_TRAINED")
