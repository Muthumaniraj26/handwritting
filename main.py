import cv2
import easyocr
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

def ocr_on_image(image_path):
    # Load and read text from image
    image = cv2.imread(image_path)
    results = reader.readtext(image)

    # Draw bounding boxes and text on image
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = tuple(map(int, tl))
        br = tuple(map(int, br))
        cv2.rectangle(image, tl, br, (0, 255, 0), 2)
        cv2.putText(image, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Prepare image for plotting
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract only the text from results
    detected_words = [text for (_, text, _) in results]

    # Show image and words side by side
    plt.figure(figsize=(12, 6))

    # Show image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Detected Letters")

    # Show extracted words
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Extracted Text")
    word_text = "\n".join(detected_words) if detected_words else "No text detected."
    plt.text(0, 1, word_text, fontsize=14, verticalalignment='top')

    plt.tight_layout()
    plt.show()


def ocr_on_camera():
    cap = cv2.VideoCapture(0)

    print("Live OCR Started. Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OCR on frame
        results = reader.readtext(frame)

        # Draw results
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
            cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Convert frame to RGB for matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        plt.axis("off")
        plt.title("Live Letter Detection")
        plt.pause(0.001)
        plt.clf()

        # To quit, check for keyboard interrupt
        import keyboard
        if keyboard.is_pressed('q'):
            break

    cap.release()
    plt.close()
# Menu
def main_menu():
    while True:
        print("\nOCR Letter Recognition System")
        print("1. Detect Letters from Uploaded Image")
        print("2. Detect Letters from Live Camera")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            root = Tk()
            root.withdraw()  # Hide main tkinter window
            image_path = filedialog.askopenfilename(title="Select Image File")
            if image_path:
                ocr_on_image(image_path)
        elif choice == '2':
            ocr_on_camera()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
