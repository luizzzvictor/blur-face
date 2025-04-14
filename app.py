import io
import os
import tempfile
import zipfile

import cv2
import numpy as np
import streamlit as st

from utils.face_blurring import anonymize_face_pixelate, anonymize_face_simple


def load_face_detector():
    """Load and return the face detector model"""
    print("[INFO] Carregando modelo de detecção facial...")
    prototxtPath = os.path.join("face_detector", "deploy.prototxt")
    weightsPath = os.path.join(
        "face_detector", "res10_300x300_ssd_iter_140000.caffemodel"
    )
    return cv2.dnn.readNet(prototxtPath, weightsPath)


def process_image(image, net, method="simples", confidence_threshold=0.5, blocks=20):
    """Process a single image and return the blurred version"""
    # Get image dimensions
    (h, w) = image.shape[:2]

    # Create a blob and detect faces
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # Get face coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract and blur face
            face = image[startY:endY, startX:endX]
            if method == "simples":
                face = anonymize_face_simple(face, factor=3.0)
            else:
                face = anonymize_face_pixelate(face, blocks=blocks)

            # Replace original face with blurred version
            image[startY:endY, startX:endX] = face

    return image


def process_batch_images(
    images, net, method="simples", confidence_threshold=0.5, blocks=20
):
    """Process multiple images and return them as a zip file"""
    # Create a BytesIO object to store the zip file
    zip_buffer = io.BytesIO()

    # Create a zip file
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for img_name, img_data in images.items():
            # Process the image
            processed_img = process_image(
                img_data.copy(),
                net,
                method=method,
                confidence_threshold=confidence_threshold,
                blocks=blocks,
            )

            # Convert to bytes
            is_success, buffer = cv2.imencode(".png", processed_img)
            if is_success:
                # Add to zip file
                zip_file.writestr(f"desfocado_{img_name}", buffer.tobytes())

    return zip_buffer.getvalue()


def main():
    st.set_page_config(page_title="Desfoque de Faces", page_icon="🎭")

    st.title("Desfoque de Faces")

    # Mode selection
    processing_mode = st.radio(
        "Modo de Processamento",
        ["Imagem Única", "Múltiplas Imagens"],
        help="Escolha entre processar uma única imagem ou várias imagens de uma vez",
    )

    # Sidebar controls
    st.sidebar.header("Configurações")
    blur_method = st.sidebar.radio(
        "Método de Desfoque",
        ["simples", "pixelado"],
        format_func=lambda x: "Simples" if x == "simples" else "Pixelado",
    )
    confidence = st.sidebar.slider("Nível de Confiança", 0.0, 1.0, 0.5)
    if blur_method == "pixelado":
        blocks = st.sidebar.slider("Blocos de Pixelização", 5, 50, 20)
    else:
        blocks = 20

    # Load face detector
    try:
        net = load_face_detector()
    except Exception as e:
        st.error(
            "Erro ao carregar o modelo de detecção facial. Verifique se a pasta face_detector existe com os arquivos do modelo."
        )
        return

    if processing_mode == "Imagem Única":
        st.write("Faça upload de uma imagem para desfocar os rostos")
        # Single file uploader
        uploaded_file = st.file_uploader(
            "Escolha uma imagem...", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            # Read and process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Show original image
            st.subheader("Imagem Original")
            st.image(image_rgb)

            if st.button("Desfocar Rostos"):
                # Process image
                with st.spinner("Processando imagem..."):
                    blurred_image = process_image(
                        image.copy(),
                        net,
                        method=blur_method,
                        confidence_threshold=confidence,
                        blocks=blocks,
                    )
                    # Convert BGR to RGB for display
                    blurred_image_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)

                    # Show result
                    st.subheader("Imagem Processada")
                    st.image(blurred_image_rgb)

                    # Add download button
                    is_success, buffer = cv2.imencode(".png", blurred_image)
                    if is_success:
                        btn = st.download_button(
                            label="Baixar imagem desfocada",
                            data=buffer.tobytes(),
                            file_name="imagem_desfocada.png",
                            mime="image/png",
                        )
    else:
        st.write("Faça upload de múltiplas imagens para desfocar os rostos")
        # Multiple files uploader
        uploaded_files = st.file_uploader(
            "Escolha as imagens...",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.write(f"Total de imagens carregadas: {len(uploaded_files)}")

            # Create a dict to store images
            images = {}

            # Show thumbnails of uploaded images
            cols = st.columns(4)
            for idx, uploaded_file in enumerate(uploaded_files):
                # Read image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                images[uploaded_file.name] = image

                # Show thumbnail
                if idx < 4:  # Show only first 4 thumbnails
                    with cols[idx]:
                        st.image(
                            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                            caption=uploaded_file.name,
                            use_container_width=True,
                        )

            if len(uploaded_files) > 4:
                st.write(f"... e mais {len(uploaded_files) - 4} imagens")

            if st.button("Processar Todas as Imagens"):
                with st.spinner(f"Processando {len(uploaded_files)} imagens..."):
                    # Process all images and get zip file
                    zip_data = process_batch_images(
                        images,
                        net,
                        method=blur_method,
                        confidence_threshold=confidence,
                        blocks=blocks,
                    )

                    # Add download button for zip file
                    st.download_button(
                        label="Baixar todas as imagens processadas",
                        data=zip_data,
                        file_name="imagens_desfocadas.zip",
                        mime="application/zip",
                    )

                    st.success(
                        f"{len(uploaded_files)} imagens processadas com sucesso!"
                    )


if __name__ == "__main__":
    main()
