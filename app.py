from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import pywt
import math
import time
import uuid

app = Flask(__name__)

# Directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to apply DCT steganography
def dct_disguise(support_image_stream, secret_image_stream, alpha=0.01):
    # Convert image streams to NumPy arrays
    support_bytes = np.asarray(bytearray(support_image_stream.read()), dtype=np.uint8)
    secret_bytes = np.asarray(bytearray(secret_image_stream.read()), dtype=np.uint8)

    # Decode images from the input streams
    support_img = cv2.imdecode(support_bytes, cv2.IMREAD_COLOR)
    secret_img = cv2.imdecode(secret_bytes, cv2.IMREAD_COLOR)

    # Ensure images have the same size
    secret_img = cv2.resize(secret_img, (support_img.shape[1], support_img.shape[0]))

    # Convert images to YCrCb color space
    support_ycrcb = cv2.cvtColor(support_img, cv2.COLOR_BGR2YCrCb)
    secret_ycrcb = cv2.cvtColor(secret_img, cv2.COLOR_BGR2YCrCb)

    # Split channels
    support_y, _, _ = cv2.split(support_ycrcb)
    secret_y, _, _ = cv2.split(secret_ycrcb)

    # Divide the support image into 8x8 blocks
    height, width = support_y.shape
    for i in range(0, height, 8):
        for j in range(0, width, 8):
            # Apply DCT to the block from support image
            block = support_y[i:i+8, j:j+8]
            block_dct = cv2.dct(np.float32(block))

            # Modify the DCT coefficients using secret image
            # Assuming alpha is small enough that secret image can be encoded without loss
            block_dct[4:, 4:] = alpha * secret_y[i:i+4, j:j+4]

            # Apply inverse DCT to the modified block
            modified_block = cv2.idct(block_dct)

            # Replace the original block with the modified one
            support_y[i:i+8, j:j+8] = np.uint8(modified_block)

    # Merge the modified Y channel with the Cr and Cb channels of the support image
    modified_image_ycrcb = cv2.merge([support_y, support_ycrcb[:, :, 1], support_ycrcb[:, :, 2]])

    # Convert the modified image back to BGR color space
    modified_image = cv2.cvtColor(modified_image_ycrcb, cv2.COLOR_YCrCb2BGR)

    return modified_image

# Function to convert text to image
def text_to_image(secret_text, target_size):
    # Create a blank white image
    image = np.ones((target_size[0], target_size[1], 3), np.uint8) * 255

    # Set the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)
    line_type = 2

    text_size, _ = cv2.getTextSize(secret_text, font, font_scale, line_type)
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] + text_size[1]) // 2

    # Put the text in the image
    cv2.putText(image, secret_text, (text_x, text_y), font, font_scale, font_color, line_type)

    return image

# Function to encode text using DCT steganography
def dct_encode_text(support_image_stream, secret_text, alpha=0.01):
    support_bytes = np.asarray(bytearray(support_image_stream.read()), dtype=np.uint8)
    support_img = cv2.imdecode(support_bytes, cv2.IMREAD_COLOR)

    secret_img = text_to_image(secret_text, support_img.shape[:2])

    return dct_disguise(support_img, secret_img, alpha)

# Function to perform LSB encoding
def lsb_encode(support_image_stream, secret_image_stream=None, secret_text=None):
    # Load the support image
    support_image = cv2.imdecode(np.frombuffer(support_image_stream.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

    if secret_image_stream:
        # If a secret image is provided, extract its pixel values
        secret_image = cv2.imdecode(np.frombuffer(secret_image_stream.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        secret_height, secret_width, _ = secret_image.shape

        # Ensure the secret image can fit within the support image
        if secret_height > support_image.shape[0] or secret_width > support_image.shape[1]:
            return None, 'Secret image is too large for the support image'

        # Loop through each pixel of the secret image and embed it into the support image
        for i in range(secret_height):
            for j in range(secret_width):
                for channel in range(3):  # RGB channels
                    support_image[i, j, channel] = (support_image[i, j, channel] & 0xFE) | ((secret_image[i, j, channel] >> 7) & 1)

    elif secret_text:
        # If secret text is provided, convert it to binary
        secret_binary = ''.join(format(ord(char), '08b') for char in secret_text)

        secret_index = 0
        for i in range(support_image.shape[0]):
            for j in range(support_image.shape[1]):
                for channel in range(3):
                    if secret_index < len(secret_binary):
                        support_image[i, j, channel] = (support_image[i, j, channel] & 0xFE) | int(secret_binary[secret_index])
                        secret_index += 1

    return support_image, None

# Function to embed data using Haar DWT steganography
def embed_haar_dwt(support_image, secret_text=None, secret_image=None):
    # Check if either secret_text or secret_image is provided
    if secret_text is None and secret_image is None:
        raise ValueError("Either 'secret_text' or 'secret_image' must be provided")

    # Load support image
    support_image = cv2.imread(support_image, cv2.IMREAD_GRAYSCALE)
    if len(support_image.shape) != 2:
        raise ValueError("Input array 'support_image' must be a two-dimensional array")

    # Haar DWT on the support image
    coeffs_support = pywt.dwt2(support_image, 'haar')
    LL_support, (LH_support, HL_support, HH_support) = coeffs_support

    if secret_text:
        # Convert text to image
        secret_image = text_to_image(secret_text, LL_support.shape[:2])

    if secret_image is None:
        raise ValueError("Either 'secret_text' or 'secret_image' must be provided")

    # Haar DWT on the secret image
    coeffs_secret = pywt.dwt2(secret_image, 'haar')
    LL_secret, _ = coeffs_secret

    # Embed the LL coefficients of the secret image into the LL coefficients of the support image
    embedded_LL = LL_support + 0.1 * LL_secret

    # Reconstruct the new coefficients
    embedded_coeffs = (embedded_LL, (LH_support, HL_support, HH_support))
    embedded_image = pywt.idwt2(embedded_coeffs, 'haar')

    # Ensure pixel values are within the valid range [0, 255]
    embedded_image = np.clip(embedded_image, 0, 255)

    # Convert to uint8 for proper image representation
    embedded_image = embedded_image.astype(np.uint8)

    return embedded_image

# Function to encode using Haar DWT steganography
def haar_dwt_encode_support_image(support_image_stream, secret_image_stream=None, secret_text=None):
    # Convert image streams to NumPy arrays
    support_bytes = np.asarray(bytearray(support_image_stream.read()), dtype=np.uint8)

    # Decode image from the input stream
    support_img = cv2.imdecode(support_bytes, cv2.IMREAD_GRAYSCALE)

    if secret_image_stream:
        secret_bytes = np.asarray(bytearray(secret_image_stream.read()), dtype=np.uint8)
        secret_img = cv2.imdecode(secret_bytes, cv2.IMREAD_GRAYSCALE)
        result_image = embed_haar_dwt(support_img, secret_image=secret_img)
    elif secret_text:
        result_image = embed_haar_dwt(support_img, secret_text=secret_text)
    else:
        return jsonify({'error': 'No secret provided'}), 400

    return result_image

# Function to calculate smoothness
def calculate_smoothness(diff):
    thresholds = [8, 16, 32, 64, 128, 255]
    classes = [0, 1, 2, 3, 4, 5]
    bit_counts = [3, 3, 4, 5, 6, 7]

    for threshold, smooth_class, bit_count in zip(thresholds, classes, bit_counts):
        if diff < threshold:
            return smooth_class, bit_count
        return classes[-1], bit_counts[-1]

# Function to calculate d_prime matrix
def calculate_d_prime(differences, smoothness_classes, bit_counts, bit_stream):
    d_prime_matrix = []

    for i, row_diff in enumerate(differences):
        row_d_prime = []
        bit_index = 0

        for j, di in enumerate(row_diff):
            smoothness_class = smoothness_classes[i][j]
            ti = bit_counts[i][j]

            lj = 0 if smoothness_class == 0 else 2 * (smoothness_class - 1) * (2 ** 3)
            uj = 7 if smoothness_class == 0 else 2 * lj - 1

            bi_binary = bit_stream[bit_index:bit_index + ti]
            bit_index += ti
            bi = int(bi_binary, 2) if bi_binary else 0

            d_prime = abs(lj + bi)
            row_d_prime.append(d_prime)

        d_prime_matrix.append(row_d_prime)

    return d_prime_matrix

# Function to calculate m matrix
def calculate_m(differences, d_prime_matrix):
    m_matrix = []

    for i, row_diff in enumerate(differences):
        row_m = []

        for j, di in enumerate(row_diff):
            d_prime = d_prime_matrix[i][j]
            m = abs(d_prime - di)
            row_m.append(m)

        m_matrix.append(row_m)

    return m_matrix

# Function to generate stego image
def generate_stego_image(image, m_matrix, differences):
    stego_image = []

    for i, row_image in enumerate(image):
        row_stego = []

        for j in range(0, len(row_image), 2):
            pi = row_image[j]
            m = m_matrix[i][j // 2]
            di = differences[i][j // 2]

            if di % 2 == 0:
                p_prime_i = pi - math.ceil(m / 2)
                p_prime_i_plus_1 = pi + math.floor(m / 2)
            else:
                p_prime_i = pi + math.ceil(m / 2)
                p_prime_i_plus_1 = pi - math.floor(m / 2)

            row_stego.extend([p_prime_i, p_prime_i_plus_1])

        stego_image.append(row_stego)

    return np.array(stego_image, dtype=np.uint8)

# Function to apply Pixel Value Differencing steganography
def apply_diff(secret_image, support_image):
    original_image = cv2.imread(support_image)
    secret_image = cv2.imread(secret_image)
    image = np.array(original_image)
    pixel_values_flat = secret_image.flatten()
    bit_stream = ''.join(['{0:08b}'.format(pixel) for pixel in pixel_values_flat])

    # Display the new differences for the secret message
    print("\nNew Differences for Secret Message:")
    differences = []  # New list to store differences
    smoothness_classes = []  # New list to store the "smoothness" classes
    bit_counts = []  # New list to store the number of bits for each difference

    # Populate differences, smoothness_classes, and bit_counts
    for ligne in image:
        row_diff = []
        row_classes = []
        row_bit_counts = []

        for i in range(0, len(ligne) - 1, 2):
            diff = abs(int(ligne[i][0]) - int(ligne[i + 1][0]))

            row_diff.append(diff)

            # Determine the "smoothness" class and the number of bits for each difference
            smooth_class, bit_count = calculate_smoothness(diff)
            row_classes.append(smooth_class)
            row_bit_counts.append(bit_count)

        differences.append(row_diff)
        smoothness_classes.append(row_classes)
        bit_counts.append(row_bit_counts)

    # Calculate the d' matrix
    d_prime_matrix = calculate_d_prime(differences, smoothness_classes, bit_counts, bit_stream)

    # Calculate the m matrix
    m_matrix = calculate_m(differences, d_prime_matrix)

    # Generate the stego image
    stego_image = generate_stego_image(image, m_matrix, differences)

    return stego_image

# Function to encode using Pixel Value Differencing steganography
def pvd_encode_support_image(support_image_stream, secret_image_stream):
    # Convert image streams to NumPy arrays
    support_bytes = np.asarray(bytearray(support_image_stream.read()), dtype=np.uint8)
    support_img = cv2.imdecode(support_bytes, cv2.IMREAD_COLOR)

    secret_bytes = np.asarray(bytearray(secret_image_stream.read()), dtype=np.uint8)
    secret_img = cv2.imdecode(secret_bytes, cv2.IMREAD_COLOR)

    # Apply Pixel Value Differencing steganography
    stego_image = apply_diff(secret_img, support_img)

    return stego_image

@app.route('/encode', methods=['POST'])
def encode_image():
    # Check if the support image is provided
    if 'support_image' not in request.files:
        return jsonify({'error': 'No support image provided'}), 400

    # Get the support image, secret image, secret text, and algorithm from the request
    support_image = request.files['support_image']
    secret_image = request.files.get('secret_image')
    secret_text = request.form.get('secret_text')
    algorithm = request.form.get('algorithm')

    # Set default alpha value for DCT algorithm
    alpha = request.form.get('alpha', 0.01, type=float)

    # Check if a valid support image is selected
    if support_image.filename == '':
        return jsonify({'error': 'No selected support image'}), 400

    # Choose the encoding algorithm based on the provided algorithm parameter
    if algorithm == 'DCT':
        # DCT encoding
        if secret_image and secret_image.filename != '':
            result_image = dct_disguise(support_image, secret_image, alpha)
        elif secret_text:
            result_image = dct_encode_text(support_image, secret_text, alpha)
        else:
            return jsonify({'error': 'No secret provided'}), 400
    elif algorithm == 'LSB':
        # LSB encoding
        result_image, error_message = lsb_encode(support_image, secret_image, secret_text)
        if error_message:
            return jsonify({'error': error_message}), 400
    elif algorithm == 'HaarDWT':
        # Haar DWT encoding
        result_image = haar_dwt_encode_support_image(support_image, secret_image, secret_text)
    elif algorithm == 'PixelValueDifferencing':
        # Pixel Value Differencing encoding
        result_image = pvd_encode_support_image(support_image, secret_image)
    else:
        return jsonify({'error': 'Invalid algorithm selected'}), 400

    # Check if the encoding was successful
    if result_image is not None:
        # Generate a unique filename based on timestamp and/or UUID
        unique_filename = str(int(time.time())) + '_' + str(uuid.uuid4()) + '.png'
        result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        # Save the encoded image
        cv2.imwrite(result_image_path, result_image)

        # Send the encoded image as a file attachment for download
        return send_file(result_image_path, as_attachment=True, download_name='encoded_image.png')
    else:
        return jsonify({'error': 'Failed to encode the secret'}), 500

@app.route('/')
def index():
    # Serve the main HTML page
    return send_from_directory('static', 'h.html')

# Set up error handler for 404 Not Found
@app.errorhandler(404)
def not_found(error):
    return "Page not found", 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

