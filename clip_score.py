import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

# Parse arguments from the command line
def parse_args():
    parser = argparse.ArgumentParser(description="Compute CLIP scores for generated images.")
    parser.add_argument("--model", type=str, required=True, help="The model identifier (e.g., 'sd-2').")
    parser.add_argument("--is_base", type=str, required=True, choices=['True', 'False'], help="Whether the model is a base model (True) or finetuned (False).")
    return parser.parse_args()

# Load CLIP model and processor
def load_clip_model():
    model_name = "openai/clip-vit-base-patch16"  # You can choose other CLIP models available
    model = CLIPModel.from_pretrained(model_name).cuda()  # Load model on GPU if available
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor

# Function to preprocess image for CLIP
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

# Function to compute the CLIP score for each image and text prompt
def compute_clip_score_for_generated_images(generated_images_dir, dataset, model, processor):
    clip_scores = []
    image_paths = sorted([os.path.join(generated_images_dir, f) for f in os.listdir(generated_images_dir) if f.endswith(('png', 'jpg', 'jpeg'))])

    # Loop through each image and its corresponding prompt
    for i, example in enumerate(dataset['train']):
        # Get the image path and corresponding text prompt
        img_path = image_paths[i]
        text_prompt = example['text']  # Assuming 'text' is the column name for text prompts

        # Preprocess the image and text prompt
        image = preprocess_image(img_path)
        inputs = processor(text=text_prompt, images=image, return_tensors="pt", padding=True).to(model.device)

        # Forward pass through CLIP to get image and text embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs['pixel_values'])
            text_features = model.get_text_features(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

        # Normalize features and calculate cosine similarity (CLIP score)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        clip_score = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())
        clip_score = np.cos(clip_score)
        clip_score = clip_score * 100
        clip_scores.append(clip_score.item())  # Extract scalar value from cosine similarity

    return clip_scores
# Function to plot the CLIP scores
def plot_clip_scores(clip_scores, args, avg_score):
    plt.figure(figsize=(10, 6))
    
    # Plot the CLIP scores as a line graph
    plt.plot(range(len(clip_scores)), clip_scores, color='skyblue', label='Individual CLIP Score')
    
    # Add a horizontal line for the average CLIP score
    plt.axhline(avg_score, color='red', linestyle='--', label=f'Average CLIP Score: {avg_score:.2f}')
    
    # Add labels and title
    plt.xlabel('Image Index')
    plt.ylabel('CLIP Score')
    plt.title(f'CLIP Scores for Generated Images - {args.model} Model')
    
    # Display the average score in the title
    plt.legend()
    
    # Save the plot as an image
    if args.is_base:
        path = os.path.join(args.model, 'clip_score_plot_base.png')
    else:
        path = os.path.join(args.model, 'clip_score_plot_finetuned.png')
    plt.savefig(path)
    
    
    # Optionally show the plot

# Main execution
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    args.is_base = args.is_base == 'True'
    # Load model and processor
    model, processor = load_clip_model()

    # Load the Hugging Face dataset
    dataset = load_dataset("Ibbi02/genai-project-dataset-without-trigger-10-test")  # Replace with your dataset ID

    # Set up the directory for generated images
    print(f"Is base model: {args.is_base}")
    if args.is_base:
        generated_images_dir = os.path.join(args.model, 'generated_images')
    else:
        generated_images_dir = os.path.join(args.model, 'generated_images_finetuned')
    print(f"Generated images directory: {generated_images_dir}")

    # Compute CLIP scores
    clip_scores = compute_clip_score_for_generated_images(generated_images_dir, dataset, model, processor)

    # Print the average CLIP score
    average_clip_score = np.mean(clip_scores)
    print(f"Average CLIP Score: {average_clip_score:.3f}")

    # Plot the CLIP scores
    plot_clip_scores(clip_scores, args, average_clip_score)
