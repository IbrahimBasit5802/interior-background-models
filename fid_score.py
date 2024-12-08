import os
import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from skimage.io import imread
from skimage.transform import resize
import argparse
from scipy.stats import entropy

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = []
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return np.asarray(images_list)

# calculate Frechet inception distance
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate FID score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# calculate Inception Score
def calculate_inception_score(model, images, splits=10):
    # get the predictions (class probabilities) for each image
    preds = model.predict(images)
    
    # calculate p(y) the marginal distribution of all predictions
    p_y = np.mean(preds, axis=0)
    
    # compute the inception score
    scores = []
    for i in range(splits):
        # split the predictions into subgroups for better estimation
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        p_yx = np.mean(part, axis=0)
        kl_div = entropy(p_yx, p_y)
        scores.append(np.exp(kl_div))
    
    # return the mean score and standard deviation across splits
    return np.mean(scores), np.std(scores)

# load images from a folder
def load_images_from_folder(folder, image_size=(299, 299)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = imread(img_path)
        img = resize(img, image_size, mode='reflect')
        if img is not None:
            images.append(img)
    return np.asarray(images)

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="FID and Inception Score Calculation")
    parser.add_argument('--model', type=str, required=True,
                        help="Specify model name (e.g., base_model, fine_tuned_model)")
    parser.add_argument("--is_base", type=str, required=True, choices=['True', 'False'], help="Whether the model is a base model (True) or finetuned (False).")
    return parser.parse_args()

# Main function to compute FID and Inception Score
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if the model is base or fine-tuned
    args.is_base = args.is_base == 'True'  # Check if the model is base
    base_path = args.model
    original_path = os.path.join(base_path, "original")
    # Set dataset paths based on the model type
    if args.is_base:
        print("Using base model. Loading original and generated images...")
        images1_folder = original_path
        images2_folder = os.path.join(base_path, "generated_images")
    else:
        print("Using fine-tuned model. Loading original and fine-tuned generated images...")
        images1_folder = original_path
        images2_folder = os.path.join(base_path, "generated_images_finetuned")

    # Load images
    print(f"Loading original images from {images1_folder}...")
    images1 = load_images_from_folder(images1_folder)
    print(f"Loading generated images from {images2_folder}...")
    images2 = load_images_from_folder(images2_folder)
    
    # Prepare the InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    
    # Convert images to float32
    images1 = images1.astype('float32')
    images2 = images2.astype('float32')
    
    # Resize images to 299x299 (InceptionV3 input size)
    print(f"Resizing images to {images1.shape[1]}x{images1.shape[2]}...")
    images1 = scale_images(images1, (299, 299, 3))
    images2 = scale_images(images2, (299, 299, 3))
    
    # Pre-process images
    images1 = preprocess_input(images1)
    images2 = preprocess_input(images2)

    # Calculate FID
    print("Calculating FID score...")
    fid = calculate_fid(model, images1, images2)
    print(f"FID: {fid:.3f}")
    
    # Calculate Inception Score for generated images
    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(model, images2)
    print(f"Inception Score: {is_mean+is_std:.3f}")

if __name__ == '__main__':
    main()
