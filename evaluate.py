import os
import subprocess
import sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses TensorFlow logs

def run_script(script_path, model, is_base):
    print(f"Running {script_path} for {model} ({'base' if is_base else 'finetuned'})...")
    sys.stdout.flush()

    # Save the current working directory
    original_working_dir = os.getcwd()

    try:
        # Ensure the script path is absolute and correct
        script_dir = os.path.dirname(os.path.abspath(script_path))

        # Change the working directory to where the script is located
        os.chdir(script_dir)

        # Set environment variable to use UTF-8 encoding for subprocess       

        # Run the script using subprocess.check_output
        output = subprocess.check_output(
            ['python', script_path, '--model', model, '--is_base', 'True' if is_base else 'False'],
            stderr=subprocess.PIPE, text=True
        )

        # Print the raw output for debugging purposes
        #print(f"Raw output from {script_path} for {model} ({'base' if is_base else 'finetuned'}):\n{output}\n")

        return output  # Return the output

    except subprocess.CalledProcessError as e:
        # Handle error if the script fails
        print(f"Error executing {script_path} for {model}: {e}")
        print(f"Error Output: {e.stderr}")
        return ""  # Return empty string in case of error

    finally:
        # Restore the original working directory
        os.chdir(original_working_dir)

# Function to extract results (e.g., FID, Inception, CLIP scores)
# Function to extract results (e.g., FID, Inception, CLIP scores)
def parse_results(output):
    scores = {'FID': None, 'Inception': None, 'CLIP': None}

    # Look for FID, Inception, and CLIP Scores in the output
    try:
        if 'FID:' in output:
            scores['FID'] = float(output.split('FID: ')[1].split('\n')[0])
        if 'Inception Score:' in output:
            scores['Inception'] = float(output.split('Inception Score: ')[1].split('\n')[0])
        if 'Average CLIP Score:' in output:
            scores['CLIP'] = float(output.split('Average CLIP Score: ')[1].split('\n')[0])
    except (IndexError, ValueError) as e:
        print(f"Error parsing results: {e}")
    
    return scores

# Function to run experiments and collect results
def run_experiment(experiment_folder, score_types, models, is_base):
    results = []
    for model in models:
        model_name = f"{model}-base" if is_base else f"{model}-finetuned"
        scores = {'Model': model_name, 'Base/Finetuned': 'Base' if is_base else 'Finetuned'}
        for score_type in score_types:
        
            # FID and Inception use the same script
            if score_type == "FID":
                file_name = "fid"  # This is the same for both FID and Inception scores
            if score_type == "CLIP":
                file_name = "clip"
            
                
            script_name = f"{file_name.lower()}_score.py"
            
            # Ensure the path is absolute and correctly formed
            script_path = os.path.join(experiment_folder, script_name)

            # Convert the script path to an absolute path if necessary
            if not os.path.isabs(script_path):
                script_path = os.path.abspath(script_path)

            # Check if the script exists
            if not os.path.exists(script_path):
                print(f"Warning: Script not found: {script_path}")
                continue  # Skip this script if not found

            # Run the script and capture the output
            output = run_script(script_path, model, is_base)
            
            # Parse the result for the specific score type
            score = parse_results(output).get(score_type)
            if score is not None:
                if score_type == "FID":
                    scores[score_type] = score
                    score = parse_results(output).get("Inception")
                    scores["Inception"] = score
                else:
                    scores[score_type] = score
            else:
                print(f"Skipping {model} ({'base' if is_base else 'finetuned'}) for {score_type} due to parsing error.")
        
        # Append the results for the model after collecting all scores
        results.append(scores)
    return results

# Function to generate results dataframe for both experiments
def generate_results():
    # Define the models and experiment folders
    models_experiment1 = ['sd-2', 'sd-v1-4', 'sdxl-1.0', 'amused-512', 'attention-gan']
    models_experiment2 = ['sd-2', 'sd-v1-4', 'sdxl-1.0', 'amused-512']
    
    experiment1_folder = 'ade_dataset'
    experiment2_folder = 'interior_dataset'
    
    score_types = ['FID' , 'CLIP']
    
    # Collect results for experiment1 (base and fine-tuned models)
    results_exp1_base = run_experiment(experiment1_folder, score_types, models_experiment1, True)
    results_exp1_finetuned = run_experiment(experiment1_folder, score_types, models_experiment1, False)

    # # Collect results for experiment2 (base and fine-tuned models)
    results_exp2_base = run_experiment(experiment2_folder, score_types, models_experiment2, True)
    results_exp2_finetuned = run_experiment(experiment2_folder, score_types, models_experiment2, False)
    
    # Combine all results into two separate DataFrames
    df_exp1 = pd.DataFrame(results_exp1_base + results_exp1_finetuned)
    df_exp2 = pd.DataFrame(results_exp2_base + results_exp2_finetuned)

    # Save to CSV
    df_exp1.to_csv('D:\\7th Sem\\Gen AI\\Project\\stable-diffusion\\ade_dataset_results.csv', index=False)
    df_exp2.to_csv('interior_dataset_results.csv', index=False)

    # Return the DataFrames if needed
    return df_exp1, df_exp2

# Run the results generation
df_exp1, df_exp2 = generate_results()
print("ADE Dataset Results:")
print(df_exp1)
print("\nInterior Dataset Results:")
print(df_exp2)
