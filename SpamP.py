import os
import re
import csv
import nltk

# Descarga de paquetes necesarios
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Establece el directorio nltk_data si es necesario
nltk.data.path.append('C:/Users/saman/nltk_data')

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paso 1: Cargar correos desde las carpetas de spam y ham
def load_emails(folder):
    emails = []
    print(f"Cargando correos desde {folder}...")
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin-1', errors='ignore') as file:
            emails.append(file.read())
    print(f"Correos cargados desde {folder}.")
    return emails

# Paso 2: Preprocesamiento de correos (limpiar texto)
def preprocess_email(email):
    email = re.sub(r'^(.*?\n)\n', '', email, flags=re.DOTALL)  # Eliminar cabeceras
    email = re.sub(r'<.*?>', '', email)  # Eliminar HTML
    tokens = word_tokenize(email.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    return ' '.join(tokens)

# Guardar correos preprocesados en un archivo CSV
def save_preprocessed_emails(filename, emails, labels):
    print(f"Guardando correos preprocesados en {filename}...")
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['email', 'label'])
        for email, label in zip(emails, labels):
            writer.writerow([email, label])
    print(f"Correos guardados en {filename}.")

# Cargar correos preprocesados desde un archivo CSV
def load_preprocessed_emails(filename):
    print(f"Cargando correos preprocesados desde {filename}...")
    emails = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Saltar la cabecera
        for row in reader:
            emails.append(row[0])
            labels.append(int(row[1]))
    print(f"Correos cargados desde {filename}.")
    return emails, labels

# Paso 3: Ejecutar todo el proceso
def main():
    print("Iniciando el proceso...")

    # Rutas a las carpetas relativas
    spam_folder_1 = r"C:\Users\saman\OneDrive\Escritorio\IA\DetectorSpam\Spam\spam"
    spam_folder_2 = r"C:\Users\saman\OneDrive\Escritorio\IA\DetectorSpam\Spam\spam_2"
    ham_folder = r"C:\Users\saman\OneDrive\Escritorio\IA\DetectorSpam\ham\hard_ham"

    # Archivo para guardar correos preprocesados
    preprocessed_file = 'preprocessed_emails.csv'

    # Verificar si el archivo preprocesado ya existe
    if not os.path.exists(preprocessed_file):
        # Cargar correos de las carpetas spam y ham
        spam_emails_1 = load_emails(spam_folder_1)
        spam_emails_2 = load_emails(spam_folder_2)
        ham_emails = load_emails(ham_folder)

        # Combinar los correos de ambas carpetas de spam
        spam_emails = spam_emails_1 + spam_emails_2

        # Etiquetas (1 para spam, 0 para ham)
        spam_labels = [1] * len(spam_emails)
        ham_labels = [0] * len(ham_emails)

        # Combinar correos y etiquetas
        all_emails = spam_emails + ham_emails
        all_labels = spam_labels + ham_labels

        print("Preprocesando correos...")
        # Preprocesar todos los correos con barra de progreso
        all_emails_cleaned = []
        for email in tqdm(all_emails, desc="Procesando correos", unit="correo"):
            cleaned_email = preprocess_email(email)
            all_emails_cleaned.append(cleaned_email)
        print("Correos preprocesados.")

        # Guardar correos preprocesados
        save_preprocessed_emails(preprocessed_file, all_emails_cleaned, all_labels)
    else:
        # Cargar correos preprocesados desde el archivo
        all_emails_cleaned, all_labels = load_preprocessed_emails(preprocessed_file)

    print("Convirtiendo correos a vectores TF-IDF...")
    # Convertir correos a vectores TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_emails_cleaned)
    print("Conversión a TF-IDF completada.")

    print("Dividiendo datos en entrenamiento y prueba...")
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.25, random_state=42)
    print("División de datos completada.")

    print("Entrenando el modelo Perceptrón...")
    # Entrenar el modelo Perceptrón
    clf = Perceptron()
    clf.fit(X_train, y_train)
    print("Modelo entrenado.")

    print("Haciendo predicciones y evaluando el modelo...")
    # Hacer predicciones y evaluar el modelo
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy * 100:.2f}%")

# Ejecutar el script completo
if __name__ == '__main__':
    main()