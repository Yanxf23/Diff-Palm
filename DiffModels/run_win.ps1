# PowerShell script for training palm model

$CHK_DIR = ".\checkpoint\cup"

# Create directory if it doesn't exist
if (-Not (Test-Path $CHK_DIR)) {
    New-Item -ItemType Directory -Path $CHK_DIR | Out-Null
}

# Set environment variable for OpenAI log directory
$env:OPENAI_LOGDIR = $CHK_DIR

# Set raw and label paths
$RAW_DIR = "C:\Users\mobil\Desktop\25spring\stylePalm\evaluation\datasets\cup_final"
$LABEL_DIR = "C:\Users\mobil\Desktop\25summer\GenPalm\Diff-Palm\cup_deployment\cup_final_label"

# Define arguments as an array
$ARGS = @(
    "--raw_dir", "$RAW_DIR",
    "--label_dir", "$LABEL_DIR",
    "--data_type", "train",
    "--include_key", "Clean",
    "--lr", "2e-4",
    "--batch_size", "64",
    "--save_interval", "10000",
    "--diffusion_steps", "1000",
    "--noise_schedule", "linear",
    "--large_size", "128",
    "--small_size", "128",
    "--in_channels", "4",
    "--out_channels", "3",
    "--num_channels", "64",
    "--num_res_blocks", "2",
    "--learn_sigma", "True",
    "--dropout", "0.2",
    "--attention_resolutions", "4",
    "--class_cond", "False"
)

# Run training
python .\palm_train.py @ARGS
