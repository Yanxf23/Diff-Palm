# Set GPU environment variable
# powershell -ExecutionPolicy Bypass -File sample_win.ps1
$env:GPUS = "0"
$env:CUDA_VISIBLE_DEVICES = $env:GPUS

# Configuration
$BATCH = 40
$K_STEP = 500
$NUM = 200
$SAMPLES = 200
$SAME_NUM = 20

# Directories
$CHK_DIR = "./output/test-large"
$env:OPENAI_LOGDIR = $CHK_DIR
$INDIR = "datasets/test-2000"
$NPZ = "$CHK_DIR/data.npz"
$OUTDIR1 = "$CHK_DIR/label"

# Create output directory if not exists
if (!(Test-Path $CHK_DIR)) {
    New-Item -ItemType Directory -Path $CHK_DIR | Out-Null
}

# Step 1: Save NPZ
python scripts/save_npz.py `
    --input "$INDIR" `
    --outdir "$OUTDIR1" `
    --outnpz "$NPZ" `
    --num $NUM `
    --same_num $SAME_NUM

# Define argument groups
$INTRA_FLAGS = "--sharing_num $SAME_NUM --sharing_step $K_STEP"
$SAMPLE_FLAGS = "--batch_size $BATCH --num_samples $SAMPLES --use_ddim False"
$MODEL_FLAGS = "--large_size 128 --small_size 128 --in_channels 4 --out_channels 3 --num_channels 64 --num_res_blocks 2 --learn_sigma True --dropout 0.1 --attention_resolutions 4 --class_cond False"
$DIFFUSION_FLAGS = "--diffusion_steps 1000 --noise_schedule linear"

# Model path
$MODEL_PATH = "checkpoint/diffusion-netpalm-scale-128/ema_0.9999.pt"

# Step 2: Sample
$CMD = "python palm_sample_intra.py " +
    "--model_path `"$MODEL_PATH`" " +
    "--base_samples `"$NPZ`" " +
    "--batch_size $BATCH " +
    "--num_samples $SAMPLES " +
    "--use_ddim False " +
    "--diffusion_steps 1000 " +
    "--noise_schedule linear " +
    "--large_size 128 " +
    "--small_size 128 " +
    "--in_channels 4 " +
    "--out_channels 3 " +
    "--num_channels 64 " +
    "--num_res_blocks 2 " +
    "--learn_sigma True " +
    "--dropout 0.1 " +
    "--attention_resolutions 4 " +
    "--class_cond False " +
    "--sharing_num $SAME_NUM " +
    "--sharing_step $K_STEP"

Invoke-Expression $CMD