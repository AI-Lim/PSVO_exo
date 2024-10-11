import gradio as gr
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from scipy import signal
from skimage.color import rgb2gray

# Fonctions de traitement d'image
def load_image(image):
    return image

def apply_negative(image):
    img_np = np.array(image)
    negative = 255 - img_np
    return Image.fromarray(negative)

def binarize_image(image, threshold):
    img_np = np.array(image.convert('L'))
    _, binary = cv2.threshold(img_np, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(binary)

def resize_image(image, width, height):
    return image.resize((width, height))

def rotate_image(image, angle):
    return image.rotate(angle)

def histogram(image):
    img_np = np.array(image)
    gray = rgb2gray(img_np)
    
    hist, _ = exposure.histogram(gray)
    plt.figure(figsize=(6, 4))
    plt.title('Histogramme en nuances de gris')
    plt.xlabel('Intensité des pixels')
    plt.ylabel('Nombre de pixels')
    plt.bar(np.arange(len(hist)), hist, width=0.5, color='gray')
    plt.tight_layout()
    plt.show()

def mean_filter(image):
    img_np = np.array(image)
    mean_filtered = cv2.blur(img_np, (5, 5))  # Filtre moyen 5x5
    return Image.fromarray(mean_filtered)

def gaussian_filter(image):
    img_np = np.array(image)
    gaussian_filtered = cv2.GaussianBlur(img_np, (5, 5), 0)  # Filtre gaussien 5x5
    return Image.fromarray(gaussian_filtered)

def contour_extract(image):
    img_np = np.array(image.convert('L'))
    kernel = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    img_contour = signal.convolve2d(img_np, kernel, boundary='symm', mode='same')
    return Image.fromarray(np.uint8(np.absolute(img_contour)))

def morph_erosion(image):
    img_np = np.array(image.convert('L'))  # Conversion en niveaux de gris
    kernel = np.ones((5, 5), np.uint8)  # Noyau 5x5 pour l'érosion
    erosion = cv2.erode(img_np, kernel, iterations=1)
    return Image.fromarray(erosion)

def morph_dilation(image):
    img_np = np.array(image.convert('L'))  # Conversion en niveaux de gris
    kernel = np.ones((5, 5), np.uint8)  # Noyau 5x5 pour la dilatation
    dilation = cv2.dilate(img_np, kernel, iterations=1)
    return Image.fromarray(dilation)

# Sauvegarder l'image sur l'ordinateur
def save_image(image):
    img_np = np.array(image)
    save_path = "image_modifiee.png"
    cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))  # Sauvegarder en format PNG
    return save_path

# Interface Gradio
def image_processing(image, operation, threshold=128, width=100, height=100, angle=0):
    if operation == "Négatif":
        return apply_negative(image)
    elif operation == "Binarisation":
        return binarize_image(image, threshold)
    elif operation == "Redimensionner":
        return resize_image(image, width, height)
    elif operation == "Rotation":
        return rotate_image(image, angle)
    elif operation == "Histogramme":
        histogram(image)  # Affichage seulement
        return image
    elif operation == "Filtre moyen":
        return mean_filter(image)
    elif operation == "Filtre gaussien":
        return gaussian_filter(image)
    elif operation == "Contours":
        return contour_extract(image)
    elif operation == "Erosion":
        return morph_erosion(image)
    elif operation == "Dilatation":
        return morph_dilation(image)
    return image

# Mise à jour dynamique de la visibilité des champs en fonction de l'opération
def update_visibility(operation):
    return {
        "threshold": operation == "Binarisation",
        "width": operation == "Redimensionner",
        "height": operation == "Redimensionner",
        "angle": operation == "Rotation"
    }

# Interface Gradio avec style
custom_css = """
#main-header {
    color: #4CAF50;  /* Vert clair */
    font-family: 'Arial', sans-serif;
    text-align: center;
}

#upload-area {
    border: 2px solid #FF9800;  /* Orange */
    background-color: #FFF3E0;
}

#result-area {
    border: 2px solid #4CAF50;
    background-color: #E8F5E9;
}

#process-button {
    background-color: #FF9800;  /* Orange */
    color: white;
    font-size: 16px;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## 🌈 PixelCrafter: Transformez vos images avec style 🎨")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Charger Image")
        operation = gr.Radio(
            ["Négatif", "Binarisation", "Redimensionner", "Rotation", "Histogramme", "Filtre moyen", "Filtre gaussien", "Contours", "Erosion", "Dilatation"],
            label="Opération", value="Négatif"
        )

    threshold = gr.Slider(0, 255, 128, label="Seuil de binarisation", visible=False)
    width = gr.Number(value=100, label="Largeur", visible=False)
    height = gr.Number(value=100, label="Hauteur", visible=False)
    angle = gr.Number(value=0, label="Angle de Rotation", visible=False)
    image_output = gr.Image(label="Image Modifiée")

    # Mise à jour de la visibilité des champs
    operation.change(update_visibility, inputs=operation, outputs=[threshold, width, height, angle])

    # Bouton d'application
    submit_button = gr.Button("Appliquer")
    submit_button.click(image_processing, inputs=[image_input, operation, threshold, width, height, angle], outputs=image_output)

    # Sauvegarde de l'image
    save_button = gr.Button("Sauvegarder l'image")
    save_button.click(save_image, inputs=image_output, outputs=gr.File(label="Télécharger l'image"))

# Lancer l'application Gradio
demo.launch()
