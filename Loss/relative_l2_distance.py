import numpy as np

def relative_l2_distance(ground_truth_scores, predicted_scores, ymax, ymin):
    """
    Compute the Relative L2-distance (R-L2).
    
    Parameters:
    ground_truth_scores (np.ndarray): Array of ground-truth scores.
    predicted_scores (np.ndarray): Array of predicted scores.
    ymax (float): The highest possible score of an action.
    ymin (float): The lowest possible score of an action.
    
    Returns:
    float: The Relative L2-distance.
    """
    # Ensure the inputs are numpy arrays
    ground_truth_scores = np.array(ground_truth_scores)
    predicted_scores = np.array(predicted_scores)
    
    # Compute the absolute differences between ground-truth and predicted scores
    abs_differences = np.abs(ground_truth_scores - predicted_scores)
    
    # Compute the R-L2 distance
    R_L2 = np.mean(abs_differences / (ymax - ymin))
    
    return R_L2
