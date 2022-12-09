﻿import os
# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2     # python -m pip install opencv-python

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.h5'

if __name__ == "__main__":
    
    # Load trained model
    model = tf.keras.models.load_model(MODEL)

    for image_name in os.listdir(TEST_DATASET_DIR):
        
        # Load the image
        image = cv2.imread(TEST_DATASET_DIR + os.path.sep + image_name, cv2.IMREAD_GRAYSCALE)
                
        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = tf.keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)
        
        # Classify the input image
        prediction = model.predict(image_data)
        
        prediction.flatten()
        
        # Find the winner class and the probability
        winner_class = np.argmax(prediction)
        winner_probability = np.max(prediction,)*100
        
        prediction[0][int(winner_class)] = -1
        
        winner_class_2 = np.argmax(prediction)
        winner_probability_2 = np.max(prediction)*100
        
        prediction[0][int(winner_class_2)] = -1
        
        winner_class_3 = np.argmax(prediction)
        winner_probability_3 = np.max(prediction)*100
        
        
        # Build the text label
        label = f"prediction = {winner_class} ({winner_probability:.2f}%)"
        label2 = f"prediction2 = {winner_class_2} ({winner_probability_2:.2f}%)"
        label3 = f"prediction3 = {winner_class_3} ({winner_probability_3:.2f}%)"
        
        # Draw the label on the image
        output_image = cv2.resize(image, (500,500))
        cv2.putText(output_image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        if(winner_probability_2 > 1):
            cv2.putText(output_image, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        
        if(winner_probability_3 > 1):
            cv2.putText(output_image, label3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        # Show the output image        
        cv2.imshow("Output", output_image)
        
        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


# Zadanie 9.2 (1p)
# Uzupełnić wyświetlany tekst na obrazie o klasy z miejsca 2 i 3, 
# o ile ich prawdopodobieństwo jest większe od 1%.


# Zadanie 9.3 (1.5p)
# Zamiast wczytywać obrazy testowe z plików, ładować je metodą mnist.load_data() z API Keras.

# Wynik: plik tekstowy z uzupełnionym kodem oraz plik graficzny z przykładowym wynikiem predykcji.