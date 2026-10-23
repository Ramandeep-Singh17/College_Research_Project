<div align="center">

Explainable Brain Tumor Classification from MRI Using Deep Learning
ResNet50 · Grad-CAM · SHAP · LIME · Groq · Streamlit
An explainable AI research project for four-class brain MRI classification, combining deep learning predictions with visual feature-attribution methods and optional natural-language explanations.
<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow&logoColor=white" alt="TensorFlow and Keras">
  <img src="https://img.shields.io/badge/Model-ResNet50-6A5ACD" alt="ResNet50">
  <img src="https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Explainability-Grad--CAM%20%7C%20SHAP%20%7C%20LIME-2E8B57" alt="Explainability methods">
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-lightgrey" alt="Research prototype">
</p>


Overview · Results · Explainability · Setup · Limitations
</div>

Important — research and educational use only. This project is not a clinically validated diagnostic system. Its predictions and explanations must not be used as a substitute for assessment by qualified medical professionals or to make medical decisions.

Table of Contents
- Overview
- Project Objectives
- Dataset
- Model Development
- Model Performance
- Explainable AI (XAI)
- Optional Groq LLM Integration
- Streamlit Application
- Project Structure
- Technology Stack
- Installation and Setup
- How to Use the Application
- Key Contributions
- Limitations
- Future Work
- References
- Conclusion
Overview
Deep learning models can learn visual patterns from medical images and classify them into categories. However, a predicted label alone does not show which image regions influenced the model's output.

This project explores an end-to-end workflow for brain MRI image classification and model interpretability. It combines a convolutional neural network (CNN), transfer learning with ResNet50, visual explanation techniques, and an optional language-model component in an interactive Streamlit application.

The system is designed to classify MRI images into four categories:
Class	Description
Glioma	One of the tumor categories represented in the dataset
Meningioma	One of the tumor categories represented in the dataset
No Tumor	Images labelled as having no tumor in the dataset
Pituitary Tumor	Images belonging to the pituitary tumor category


Users can upload an MRI image, review the model's output scores, inspect Grad-CAM visualizations, explore SHAP and LIME explanations, and optionally request a natural-language summary generated through the Groq API.
Project at a glance
Component	Details
Task	Four-class brain MRI image classification
Primary model	Fine-tuned ResNet50
Baseline	Custom CNN
Explainability methods	Grad-CAM, SHAP, LIME
Web interface	Streamlit
Optional language model	Groq API
Held-out test images	1,584
Best reported validation accuracy	88.95%
Reported test accuracy	85.35%
Intended use	Research and education


Project Objectives
The main objectives of this project are to:
1. Develop a deep learning model for four-class brain MRI classification.
2. Compare a custom CNN baseline with frozen and fine-tuned ResNet50 transfer-learning configurations.
3. Evaluate the selected model on a separate held-out test set.
4. Visualize image regions associated with predictions using Grad-CAM.
5. Explore complementary feature-attribution methods using SHAP and LIME.
6. Use an optional LLM component to turn model-output information into readable text.
7. Build a Streamlit application that brings classification and explainability into one interface.
8. Document the limitations of model predictions and AI-generated explanations.
Dataset
Source
The project uses the publicly available Brain Tumor MRI Dataset uploaded by Masoud Nickparvar on Kaggle.
- Dataset: Brain Tumor MRI Dataset
- Uploader: Masoud Nickparvar
- Source: View the dataset on Kaggle
The images are organized into four classes: glioma, meningioma, no tumor, and pituitary tumor.
Dataset split
The dataset used in the experiments contained 7,013 images.
Subset	Images	Purpose
Original training folder	5,429	Split into training and validation subsets
Training subset	4,343	Model fitting
Validation subset	1,086	Model selection and validation monitoring
Separate test folder	1,584	Held-out evaluation
Total	7,013	


The original training folder was split into training and validation subsets using an 80:20 stratified split with random_state=42. The separate test folder was kept apart from training and validation for final evaluation.
Original class distribution
Class	Original training folder	Test folder
Glioma	1,400	386
Meningioma	1,386	398
No Tumor	1,281	400
Pituitary	1,362	400
Total	5,429	1,584


These counts describe the original folders. Exact per-class counts after the stratified training-validation split should be obtained from the data-splitting code if needed for publication.
Model Development
1. Custom CNN Baseline
A custom convolutional neural network was used as an initial baseline.

CNNs learn spatial features from images through convolutional filters, nonlinear activations, and pooling operations. The learned features are then passed to classification layers to produce class scores.
Metric	Result
Training accuracy	83.21%
Final validation accuracy	85.64%
Best validation accuracy	86.00%
Best validation loss	0.3732


The original custom CNN training history was retained as a JSON file. A reconstructed custom CNN baseline was also saved for further experiments. Since the original model weights were not retained, the reconstructed baseline should not be assumed to be identical to the original training run.
2. Frozen ResNet50
ResNet50 is a deep CNN architecture that uses residual connections to help train deeper networks. Transfer learning adapts representations learned by a pretrained model to a new task.

In the frozen configuration, the pretrained ResNet50 backbone was kept frozen while the classification head was trained.
Metric	Result
Validation accuracy	80.85%


3. Fine-tuned ResNet50
In the fine-tuned configuration, selected ResNet50 backbone layers were unfrozen and trained on the brain MRI dataset.

Reported configuration details:
- Optimizer: Adam
- Learning rate: 1e-5
- Batch Normalization: Layers kept frozen
- Best validation accuracy: 88.95%
- Validation loss at the reported best epoch: 0.3200
Primary saved model:
models/resnet50_finetuned.keras
The exact preprocessing configuration, input resolution, and class-index mapping must match the original training pipeline. Verify these settings before using the application for formal evaluation.
Model comparison
Model	Validation accuracy
Custom CNN baseline	86.00%
Frozen ResNet50	80.85%
Fine-tuned ResNet50	88.95%


Among these evaluated configurations, the fine-tuned ResNet50 achieved the highest validation accuracy. Validation performance was used for model comparison, while the separate test set was used to estimate final classification performance.
Model Performance
The fine-tuned ResNet50 was evaluated on a held-out test set containing 1,584 images.
Overall test metrics
Metric	Test result
Accuracy	85.35%
Macro precision	85.35%
Macro recall	85.23%
Macro F1-score	85.13%
Weighted precision	85.41%
Weighted recall	85.35%
Weighted F1-score	85.23%


Macro metrics calculate a score independently for each class and then average the class scores. Weighted metrics account for the number of examples in each class.
Class-wise test performance
Class	Precision	Recall	F1-score	Support
Glioma	79.49%	72.28%	75.71%	386
Meningioma	78.62%	80.40%	79.50%	398
No Tumor	86.50%	97.75%	91.78%	400
Pituitary	96.79%	90.50%	93.54%	400


The model performed more strongly on the no-tumor and pituitary classes than on glioma and meningioma. The lower glioma recall indicates that some glioma images were assigned to other classes.

These results show why class-wise metrics matter: overall accuracy alone does not describe how a model performs across all categories. The reported metrics apply to this dataset and do not establish clinical performance on images from other hospitals, scanners, or patient populations.
Explainable AI (XAI)
Explainable AI methods can help researchers inspect which parts of an input may influence a model's output. This project explores Grad-CAM, SHAP, and LIME, which provide different types of explanations.

These techniques are not interchangeable, and none independently proves that a prediction is correct or clinically meaningful.
Grad-CAM
Gradient-weighted Class Activation Mapping (Grad-CAM) uses gradients associated with a selected class and feature maps from a convolutional layer to create a heatmap.

The application is designed to display:
- The original MRI image
- The Grad-CAM heatmap
- The heatmap overlaid on the original image
Warm regions indicate stronger attribution under the selected visualization settings. They do not necessarily identify exact tumor boundaries or prove that the model relied on medically meaningful features.

The convolutional layer used for Grad-CAM must be verified against the saved model's architecture.
SHAP
SHAP (SHapley Additive exPlanations) is based on Shapley values from cooperative game theory. It estimates how input features contribute to a model output relative to a reference or background.

For image classification, SHAP may attribute contributions to pixels or image regions depending on the selected explainer and representation.

In this project, SHAP is intended to provide a complementary explanation of the selected model prediction. Its implementation must be compatible with the actual TensorFlow/Keras model and preprocessing pipeline.

Important considerations:
- Image explainers may require background samples and masking operations.
- Repeated model evaluations can increase runtime and memory usage.
- A SHAP attribution map is not a tumor segmentation mask.
- Interpretation depends on the explainer, reference data, masking strategy, and selected output class.
LIME
LIME (Local Interpretable Model-agnostic Explanations) approximates model behavior around one input using a locally interpretable model.

For image explanations, LIME typically:
1. Divides the image into superpixels.
2. Generates perturbed versions by hiding or modifying selected regions.
3. Queries the trained model for predictions on those samples.
4. Fits a local interpretable approximation.
5. Displays superpixels that support or oppose the selected class under the chosen settings.
LIME is model-agnostic, but its results can depend on superpixel segmentation, sample count, random seed, and prediction function. The installed LIME API must also match the arguments used in the application.

The integration should be considered verified only after the computation completes successfully and produces an actual explanation.
Comparing the explanation methods
Method	Main purpose	Explanation type
Grad-CAM	Highlight regions associated with a selected class	Gradient-based heatmap
SHAP	Estimate feature contributions relative to a reference	Feature-attribution values
LIME	Approximate behavior around one input	Local surrogate and superpixel explanation


Using multiple methods can help researchers inspect model behavior from different perspectives. Agreement between explanation maps does not automatically prove that the model is correct or that an explanation is clinically valid.
Optional Groq LLM Integration
Why use an LLM?
Heatmaps and numerical class scores can be difficult to interpret without additional context. An optional large language model (LLM) component uses the Groq API to generate a readable natural-language summary based on the information supplied by the application.
Intended workflow
1. The image classifier generates class scores.
2. The user reviews the predicted class.
3. The application prepares relevant prediction and explanation information.
4. The user requests a textual explanation.
5. The application sends the configured prompt to the Groq model.
6. The returned text is displayed in the Streamlit interface.
The LLM does not replace the image classifier. Its output may be incomplete or inaccurate and should be treated as a research-oriented summary rather than an independently validated medical interpretation.
Configure the Groq API key
Create this file locally:
.streamlit/secrets.toml
Add your API key:
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
The initial configuration used openai/gpt-oss-120b. Model availability and account access should be checked if a request fails.

The Groq integration is optional. Core image prediction and local explanation methods should remain usable when the API is unavailable.
Security: Never commit your actual API key or .streamlit/secrets.toml file to GitHub.

Streamlit Application
The project includes a Streamlit interface intended to combine classification, explainability, and evaluation information in one place.
Main interface features
- MRI image upload
- Predicted class and four-class output scores
- Grad-CAM heatmap and overlay
- SHAP explanation panel
- LIME explanation panel
- Comparison of explanation methods
- Optional Groq-generated textual explanation
- Model evaluation metrics
- Responsible-use and limitation information
The model and explanation methods should use a consistent preprocessing pipeline and class-index mapping. The application also exposes settings for the computational cost of SHAP and LIME. More explainer samples or evaluations may provide additional information but can increase runtime and memory use.
Implementation notes
The local application recognized the Groq API key. During development, the LIME integration encountered an API compatibility error because the installed LimeImageExplainer.explain_instance() method did not accept the progress_bar keyword argument; removal of the incompatible argument was identified as the correction.

SHAP and LIME should be described as successfully integrated only after their computations and visualizations have been verified. The model's preprocessing and class-index mapping also need to be checked against the original training pipeline before relying on predictions.
Project Structure
College_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── resnet50_finetuned.keras
│   ├── custom_cnn_baseline.keras
│   ├── resnet50_test_results.json
│   ├── model_comparison_test_results.json
│   └── custom_cnn_history.json
├── .streamlit/
│   └── secrets.toml
└── .venv/
- app.py — Streamlit application entry point.
- requirements.txt — Python dependencies required by the application.
- models/resnet50_finetuned.keras — Primary fine-tuned model.
- models/custom_cnn_baseline.keras — Saved reconstructed CNN baseline.
- models/resnet50_test_results.json — Stored test-result data.
- models/model_comparison_test_results.json — Stored model-comparison results.
- models/custom_cnn_history.json — Custom CNN training history.
- .streamlit/secrets.toml — Local secrets configuration; do not commit it.
- .venv/ — Local Python virtual environment; do not upload it.
The actual paths used by app.py must match the repository structure. If the code expects files in different locations, update the paths consistently.
Technology Stack
Technology	Role in the project
Python	Main programming language
TensorFlow / Keras	Model loading and inference
CNN	Baseline image classification
ResNet50	Transfer learning and fine-tuning
NumPy	Numerical computation
Pillow	Image loading and processing
OpenCV	Image-processing utilities, where used
Matplotlib	Visualization
scikit-learn	Data splitting and evaluation metrics
Grad-CAM	Gradient-based visual explanations
SHAP	Feature-attribution explanations
LIME	Local superpixel-based explanations
Streamlit	Interactive web interface
Groq API	Optional LLM-generated textual explanations
Plotly	Interactive charts, where used


Installation and Setup
Prerequisites
- Python 3.11.9 was used for the local setup.
- Visual Studio Code or another Python IDE.
- The saved model and JSON files listed in the project structure.
- Internet access for package installation and optional Groq API requests.
Step 1: Open the project
Open the College_Project directory in VS Code and open a terminal in that directory.
Step 2: Create a virtual environment
In PowerShell, run:
python -m venv .venv
Activate it:
.\.venv\Scripts\Activate.ps1
Confirm that (.venv) appears in the terminal.
Step 3: Install dependencies
Ensure requirements.txt contains the packages used by the application. The local setup may require:
streamlit
tensorflow
numpy
pillow
matplotlib
pandas
scikit-learn
opencv-python-headless
grad-cam
shap
lime
groq
plotly
scikit-image
Install the dependencies:
python -m pip install --upgrade pip
pip install -r requirements.txt
Package versions should be pinned after a compatible working environment has been tested. The presence of a package does not guarantee API compatibility with every other installed package.
Step 4: Verify the model files
Confirm that the primary model exists at:
models/resnet50_finetuned.keras
Also verify the paths for the JSON files and optional baseline model. Update the paths in app.py if the actual directory structure differs.
Step 5: Configure Groq (optional)
Create .streamlit/secrets.toml and add the API key as described in the Groq integration section.

Add these entries to .gitignore:
.venv/
__pycache__/
.streamlit/secrets.toml
Never commit or share the actual API key.
Step 6: Run the application
streamlit run app.py
Streamlit will display a local URL in the terminal. Open that URL in your browser.
How to Use the Application
1. Launch the Streamlit application.
2. Upload a supported MRI image.
3. Review the predicted class and output scores.
4. Open the Grad-CAM panel to inspect the heatmap and overlay.
5. Run SHAP to generate a feature-attribution explanation.
6. Run LIME to generate a local superpixel-based explanation.
7. Compare the explanation methods in the interface.
8. Optionally request a Groq-generated textual explanation.
9. Review the evaluation metrics and limitations.
For controlled testing, use an image with a known dataset label. A successful upload or prediction does not, by itself, verify that preprocessing and class labels are correct. Confirm the model configuration before interpreting the results.
Key Contributions
This project brings several components together in a single experimental workflow:
- Four-class brain MRI classification using a fine-tuned ResNet50.
- Comparison of a custom CNN, frozen ResNet50, and fine-tuned ResNet50.
- Evaluation on a separate held-out test set with class-wise metrics.
- Grad-CAM visualization of regions associated with model predictions.
- SHAP and LIME as complementary explanation methods, subject to successful execution and verification.
- Optional LLM-generated text through the Groq API.
- A Streamlit interface that presents predictions, visual explanations, and evaluation information together.
The main contribution is the integration of image classification, interpretability, and optional natural-language explanation in one research-oriented application. The project does not claim to establish a new clinical diagnostic method.
Limitations
1. Limited dataset scope: The model was evaluated on one dataset; generalization to external datasets has not been established.
2. Uneven class performance: Test metrics differ across the four classes.
3. Challenging classes: Glioma and meningioma have lower reported performance than the no-tumor and pituitary classes.
4. Heatmap interpretation: Grad-CAM does not establish exact tumor boundaries.
5. Explainer sensitivity: SHAP and LIME outputs depend on their settings and require stability and usefulness checks.
6. LLM reliability: Generated text may contain unsupported or inaccurate statements.
7. Probability calibration: Output scores are not necessarily calibrated clinical probabilities.
8. Preprocessing consistency: The preprocessing pipeline and class-index mapping must be verified against the original training pipeline.
9. No external clinical validation: External clinical validation and prospective evaluation have not been established.
10. Not for diagnosis: The application is not intended for clinical diagnosis or treatment decisions.
Future Work
- Verify and document the exact preprocessing pipeline and class-index mapping.
- Resolve and test SHAP and LIME compatibility issues.
- Evaluate explanation stability across repeated runs and different images.
- Investigate misclassifications between glioma and meningioma.
- Evaluate the model on independent MRI datasets.
- Compare explanation methods quantitatively and qualitatively.
- Investigate probability calibration and uncertainty estimation.
- Improve inference efficiency and caching in the Streamlit application.
- Assess the faithfulness and factual reliability of LLM-generated explanations.
- Explore segmentation-based methods if precise tumor localization is required.
References
1. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). https://doi.org/10.1109/CVPR.2016.90
2. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. Proceedings of the IEEE International Conference on Computer Vision (ICCV). https://doi.org/10.1109/ICCV.2017.74
3. Lundberg, S. M., and Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. Advances in Neural Information Processing Systems. Paper
4. Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). “Why Should I Trust You?” Explaining the Predictions of Any Classifier. Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939778
5. Nickparvar, M. Brain Tumor MRI Dataset. Kaggle. Dataset page
Conclusion
This project investigates multi-class brain MRI classification using CNN-based approaches and transfer learning with ResNet50. The fine-tuned ResNet50 achieved 88.95% validation accuracy and 85.35% accuracy on the held-out test set, with a macro-F1 score of 85.13%.

Grad-CAM provides a way to visually inspect model predictions, while SHAP and LIME offer complementary feature-attribution approaches. An optional Groq-powered LLM component generates readable summaries of model output.

Overall, the project demonstrates an experimental workflow that combines image classification, interpretability, and natural-language explanation in an interactive Streamlit application. Further testing, external validation, and systematic evaluation of explanation reliability are required before making stronger claims about robustness or clinical usefulness.
