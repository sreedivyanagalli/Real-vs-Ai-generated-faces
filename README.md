-Real vs AI-Generated Face Classification-

-Project Objective-
This project aims to distinguish between real human faces and AI-generated (deepfake or GAN-generated) faces using deep learning. It uses convolutional neural networks (CNNs) trained on image datasets to classify whether a given face is authentic or synthetic.

-Tech Stack-
- Python
- TensorFlow / Keras
- CNN (Convolutional Neural Network)
- OpenCV / PIL (image processing)
- Matplotlib / Seaborn (visualization)
- Jupyter Notebook

-Dataset-
- Real Faces: Scraped or sourced from public datasets
- AI Faces: Downloaded from sites like `thispersondoesnotexist.com` or GANFace datasets
- All images are resized, normalized, and split into:
  - `train/`, `validation/`, `test/` directories

-Model Architecture-
- Input Layer (Resized face images)
- Convolutional layers + MaxPooling
- Flatten + Dense Layers
- Output: Binary classifier (Real vs Fake)

Loss Function: Binary Cross-Entropy  
Optimizer: Adam  
Metrics: Accuracy, Precision, Recall

-Results-
- Validation Accuracy: ~85–90%
- The model generalizes well to unseen faces
- Example Predictions: Heatmaps / Class Activation Maps (optional)

-How to Use-
1. Clone the repository:
   ```bash
   git clone https://github.com/sreedivyanagalli/Real-vs-Ai-generated-faces.git
   cd Real-vs-Ai-generated-faces
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook real_vs_ai_face_classifier.ipynb
   ```
