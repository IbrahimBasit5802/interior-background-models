from datasets import load_dataset
from PIL import Image
import os
import torch
from diffusers import AmusedPipeline
from safetensors.torch import load_file  # Import safetensors for loading weights

# Load the dataset
dataset = load_dataset("Ibbi02/genai-project-dataset-without-trigger-10-test")

# Create directories for saving images
generated_dir = "./generated_images_finetuned"
real_dir = "./original"
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(real_dir, exist_ok=True)

# Load your base Amused model
model_name = "amused/amused-512"  # Update this to the path of your base model
pipe = AmusedPipeline.from_pretrained(
    model_name, variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Load LoRA weights
lora_weights_path = "amuse-512-finetuned\pytorch_lora_weights.safetensors"  # Path to your LoRA weights

# Load LoRA weights using safetensors
lora_weights = load_file(lora_weights_path)  # Load the weights from safetensors file

# Apply LoRA weights to the transformer (or other layers)
def apply_lora_weights(model, lora_weights):
    for name, param in model.named_parameters():
        if name in lora_weights:
            param.data = param.data + lora_weights[name].data  # Adjust this as per your LoRA integration method

apply_lora_weights(pipe.transformer, lora_weights)  # Apply LoRA to the transformer, adjust if necessary

# Define prompts and negative prompt
negative_prompt = "low quality, ugly"

# Process dataset and generate images
for idx, item in enumerate(dataset["train"]):  # Adjust if there's a test split
    prompt = item["text"]  # Column for text prompts

    # Generate image using the fine-tuned model
    generator = torch.manual_seed(0)
    generated_image = pipe(prompt, negative_prompt=negative_prompt, generator=generator).images[0]

    # Save both real and generated images
    generated_image.save(os.path.join(generated_dir, f"generated_{idx}.png"))

print(f"Generated images saved in {generated_dir}")
