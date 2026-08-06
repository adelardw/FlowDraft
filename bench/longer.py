"""Догоняет ли многокомпонентный лосс одно-компонентный при большем бюджете?

Гипотеза: обычный Orthrus выигрывает на 2000 шагах потому, что его цель
одно-компонентная и сходится быстрее, а `orthrus_ms` (та же цель плюс член
самокоррекции) просто не доучен.

Проверяется как ТРАЕКТОРИЯ, а не одной точкой: обе руки продолжаются с
имеющихся 2000 до 6000 со снимками на 4000 и 6000. Одной точкой «догнал бы»
неотличимо от шума; двумя видно, сходится разрыв или держится.

Обе руки продолжаются, а не только `orthrus_ms` — иначе разница по числу шагов
смешается с разницей по составу лосса.

  uv run python .work/longer.py
"""
import os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TARGET = int(os.environ.get("TARGET", 6000))
SRC = "checkpoints/final_ab"
ROOT = "/tmp/fd-long"

COMMON = [
    "model=smollm2_135m", "train.block_size=8", "train.anchors_per_sequence=1",
    "train.lr=3e-4", "train.val_decode_prompts=0", "train.early_stop_patience=0",
    "train.monitor=val/loss", "train.monitor_mode=min",
    "data.batch_size=2", "data.max_length=256", "data.val_size=8",
    "data.shuffle_buffer=64",
    f"trainer.max_steps={TARGET}", f"trainer.val_check_interval={TARGET}",
    "trainer.precision=32", "trainer.log_every_n_steps=200",
    # Снимки внутри прогона: они и есть точки траектории.
    "train.checkpoint_every_n_steps=2000",
    # Кавычки обязательны: грамматика оверрайдов Hydra спотыкается на '{'.
    "train.checkpoint_name='snap-{step:07d}'",
]

PW = ["train.teacher_chain_tail_weight=0.3",
      "train.position_weights=[2.32,1.75,1.41,0.87,0.39,0.19,0.07]"]
FD = ["train.variant=flowdraft_block_wise", "train.prior_type=discunif",
      "train.verify_kl_weight=1.0", "train.endpoint_weight=0.0", "train.lambda=0.0",
      "train.terminal_time_fraction=1.0", *PW]

ARMS = {
    "orthrus": ["train.variant=orthrus"],
    "orthrus_ms": ["train.variant=orthrus", "train.selfcorrect_kl_weight=1.0",
                   "train.selfcorrect_tail_weight=0.5"],
    # Симплексные руки на том же горизонте: сравнение состояний на 2000 шагах
    # было бы повтором той же ошибки, из-за которой разрыв выглядел реальным.
    "fd_base": [*FD],
    "fd_ms": [*FD, "train.selfcorrect_kl_weight=1.0", "train.selfcorrect_rounds=2",
              "train.selfcorrect_s_min=0.0", "train.selfcorrect_tail_weight=0.5"],
    # Четырёхкомпонентная рука: из всех пострадала от короткого горизонта
    # сильнее всего, плюс её рампа в 500 шагов означала, что на прогоне в 2000
    # веса CFM стояли на полном уровне лишь три четверти времени.
    "fd_cfm_ms": ["train.variant=flowdraft_block_wise", "train.prior_type=discunif",
                  "train.verify_kl_weight=1.0", *PW,
                  "train.endpoint_weight=0.5", "train.lambda=0.25",
                  "train.terminal_time_fraction=0.25", "train.lambda_ramp_steps=500",
                  "train.selfcorrect_kl_weight=1.0", "train.selfcorrect_rounds=2",
                  "train.selfcorrect_s_min=0.0", "train.selfcorrect_tail_weight=0.5"],
}


def steps_of(path):
    if not os.path.exists(path):
        return 0
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("global_step", 0))


for name, extra in ARMS.items():
    out = f"{ROOT}/{name}"
    if steps_of(f"{out}/last.ckpt") >= TARGET:
        print(f"[{name}] уже {TARGET}", flush=True)
        continue
    resume = f"{SRC}/{name}/last.ckpt"
    print(f"[{name}] продолжаю с {steps_of(resume)} до {TARGET}...", flush=True)
    log = f"{out}.trainlog"
    os.makedirs(ROOT, exist_ok=True)
    with open(log, "w") as sink:
        proc = subprocess.Popen(
            [sys.executable, "src/train.py", *COMMON, *extra,
             f"output_dir={out}", f"resume_from_checkpoint={resume}"],
            stdout=sink, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + 30 * 60 + TARGET * 2
    while True:
        # Процесс виснет в деструкторе pyarrow ПОСЛЕ завершения, поэтому
        # готовность читается из чекпоинта, а не из кода возврата.
        if steps_of(f"{out}/last.ckpt") >= TARGET:
            proc.kill(); proc.wait(timeout=30); break
        if proc.poll() is not None:
            break
        if time.time() > deadline:
            proc.kill(); print(f"[{name}] таймаут", flush=True); break
        time.sleep(20)
    got = steps_of(f"{out}/last.ckpt")
    if got < TARGET:
        print(open(log).read()[-2500:])
        raise SystemExit(f"[{name}] ПРОВАЛ: {got} шагов из {TARGET}")
    print(f"[{name}] готово, {got} шагов", flush=True)
print("TRAINED_LONG")
