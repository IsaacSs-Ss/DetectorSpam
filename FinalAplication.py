import os
import re
import pickle
import email
import imaplib
from email.header import decode_header
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
import shutil
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Preprocesamiento de correos electrónicos
def preprocess_email(email_content):
    email_content = re.sub(r'<.*?>', '', email_content)  # Eliminar HTML
    tokens = word_tokenize(email_content.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
    return ' '.join(tokens)

# Cargar correos de una carpeta
def load_emails(folder):
    emails = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
                emails.append(file.read())
    return emails

# Función para entrenar el modelo
def train_model():
    # Ruta base del directorio donde se encuentra el archivo de código
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Rutas relativas para las carpetas de spam y ham
    spam_folder_1 = os.path.join(base_dir, "data", "Spam", "spam")
    spam_folder_2 = os.path.join(base_dir, "data", "Spam", "spam_2")
    ham_folder = os.path.join(base_dir, "data", "ham", "hard_ham")
    model_dir = os.path.join(base_dir, "models")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    clf_model_path = os.path.join(model_dir, "clf_model.pkl")


    # Crear las carpetas de spam y ham si no existen
    os.makedirs(spam_folder_1, exist_ok=True)
    os.makedirs(spam_folder_2, exist_ok=True)
    os.makedirs(ham_folder, exist_ok=True)

    spam_emails = load_emails(spam_folder_1) + load_emails(spam_folder_2)
    ham_emails = load_emails(ham_folder)
    all_emails = spam_emails + ham_emails
    all_labels = [1] * len(spam_emails) + [0] * len(ham_emails)
    all_emails_cleaned = [preprocess_email(email) for email in tqdm(all_emails, desc="Preprocesando correos")]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_emails_cleaned)
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.25, random_state=42)
    clf = Perceptron()
    clf.fit(X_train, y_train)

    with open("vectorizer.pkl", "wb") as vec_file:
        pickle.dump(vectorizer, vec_file)
    with open("clf_model.pkl", "wb") as model_file:
        pickle.dump(clf, model_file)

    accuracy = accuracy_score(y_test, clf.predict(X_test))
    messagebox.showinfo("Entrenamiento", f"Modelo entrenado con precisión del {accuracy * 100:.2f}%")

# Función para agregar más datos de entrenamiento
def add_data():
    # Ruta base del directorio donde se encuentra el archivo de código
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Rutas relativas para las carpetas de spam y ham
    spam_folder_1 = os.path.join(base_dir, "Spam", "spam")
    ham_folder = os.path.join(base_dir, "ham", "hard_ham")

    # Crear las carpetas de spam y ham si no existen
    os.makedirs(spam_folder_1, exist_ok=True)
    os.makedirs(ham_folder, exist_ok=True)

    # Preguntar al usuario si los datos son spam o ham
    data_type = messagebox.askquestion("Tipo de Datos", "¿Son datos de Spam? Presiona 'Yes' para Spam y 'No' para Ham.")
    target_folder = spam_folder_1 if data_type == 'yes' else ham_folder
    
    # Seleccionar carpeta de correos para agregar
    folder_selected = filedialog.askdirectory(title="Selecciona la carpeta de correos")
    if not folder_selected:
        messagebox.showinfo("Cancelado", "No se seleccionó ninguna carpeta.")
        return
    
    # Copiar los archivos seleccionados a la carpeta de destino (spam o ham)
    for filename in os.listdir(folder_selected):
        src_path = os.path.join(folder_selected, filename)
        dest_path = os.path.join(target_folder, filename)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
    
    messagebox.showinfo("Datos Agregados", f"Los datos se han agregado a la carpeta {'Spam' if data_type == 'yes' else 'Ham'}.")

# Cargar modelo y vectorizador
def load_model():
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    with open("clf_model.pkl", "rb") as model_file:
        clf = pickle.load(model_file)
    return vectorizer, clf

# Clasificar un solo correo ingresado manualmente
def classify_single_email():
    vectorizer, clf = load_model()
    email_text = entry_email.get("1.0", tk.END)
    email_cleaned = preprocess_email(email_text)
    email_vectorized = vectorizer.transform([email_cleaned])
    is_spam = clf.predict(email_vectorized)[0]

    result = "Spam" if is_spam else "No es Spam"
    messagebox.showinfo("Resultado", f"El correo ingresado es: {result}")

# Clasificar todos los correos de la bandeja de entrada
def classify_all_emails():
    username = entry_username.get()
    password = entry_password.get()

    if not username or not password:
        messagebox.showerror("Error", "Debes ingresar tu correo y contraseña.")
        return

    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    try:
        imap.login(username, password)
        imap.select("INBOX")

        vectorizer, clf = load_model()
        status, messages = imap.search(None, "ALL")
        results = []

        # Configurar la barra de progreso
        message_ids = messages[0].split()
        total_messages = len(message_ids)
        progress["maximum"] = total_messages

        for idx, num in enumerate(message_ids, start=1):
            # Actualizar la barra de progreso
            progress["value"] = idx
            root.update_idletasks()

            status, msg_data = imap.fetch(num, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    email_content = ""

                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                try:
                                    email_content += part.get_payload(decode=True).decode()
                                except UnicodeDecodeError:
                                    email_content += part.get_payload(decode=True).decode('latin-1')
                    else:
                        try:
                            email_content = msg.get_payload(decode=True).decode()
                        except UnicodeDecodeError:
                            email_content = msg.get_payload(decode=True).decode('latin-1')

                    email_cleaned = preprocess_email(email_content)
                    email_vectorized = vectorizer.transform([email_cleaned])
                    is_spam = clf.predict(email_vectorized)[0]
                    status = "Spam" if is_spam else "No es Spam"
                    
                    # Decodificar el asunto del correo
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        try:
                            subject = subject.decode('utf-8')
                        except UnicodeDecodeError:
                            subject = subject.decode('latin-1')

                    # Agregar tupla con el asunto y el estado
                    results.append((subject, status))

        display_result(results if results else [("No hay correos en la bandeja.", "")])
    except imaplib.IMAP4.error:
        messagebox.showerror("Error", "Falló el inicio de sesión. Verifica tus credenciales.")
    finally:
        imap.logout()
        progress["value"] = 0  # Restablecer la barra de progreso

# Configuración de la interfaz principal con Tkinter
root = tk.Tk()
root.title("Detector de Spam")
root.geometry("700x750")

frame = tk.Frame(root)
frame.pack(pady=20)

title = tk.Label(frame, text="Detector de Spam", font=("Arial", 18))
title.pack(pady=10)

# Frame para los botones de entrenamiento y agregar datos
button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

# Botón para entrenar el modelo
train_button = tk.Button(button_frame, text="Entrenar Modelo", command=train_model)
train_button.grid(row=0, column=0, padx=10)

# Botón para agregar más datos de entrenamiento
add_data_button = tk.Button(button_frame, text="Agregar Datos", command=add_data)
add_data_button.grid(row=0, column=1, padx=10)

# Opciones de análisis individual
label_email = tk.Label(frame, text="Ingresa el contenido de un correo para clasificar:")
label_email.pack(pady=5)

entry_email = tk.Text(frame, height=10, width=60)
entry_email.pack(pady=5)

classify_button = tk.Button(frame, text="Clasificar Correo Individual", command=classify_single_email)
classify_button.pack(pady=5)

# Separador visual
separator = tk.Label(frame, text="----------------------")
separator.pack(pady=10)

# Opciones de análisis de toda la bandeja de entrada
label_inbox_analysis = tk.Label(frame, text="Análisis de toda la bandeja de entrada:")
label_inbox_analysis.pack(pady=5)

label_username = tk.Label(frame, text="Correo Electrónico:")
label_username.pack(pady=5)
entry_username = tk.Entry(frame, width=30)
entry_username.pack(pady=5)

label_password = tk.Label(frame, text="Contraseña:")
label_password.pack(pady=5)
entry_password = tk.Entry(frame, show="*", width=30)
entry_password.pack(pady=5)

classify_all_button = tk.Button(frame, text="Clasificar Todos los Correos", command=classify_all_emails)
classify_all_button.pack(pady=5)

# Tabla para mostrar los resultados
results_label = tk.Label(frame, text="Resultados:")
results_label.pack(pady=5)

# Configuración del Treeview para mostrar la tabla
columns = ("Asunto", "Estado")
results_table = ttk.Treeview(frame, columns=columns, show="headings", height=15)
results_table.heading("Asunto", text="Asunto del Correo")
results_table.heading("Estado", text="Estado")

# Ajuste de ancho de columnas
results_table.column("Asunto", width=400)
results_table.column("Estado", width=100)

results_table.pack(pady=5)

# Barra de progreso
progress_label = tk.Label(frame, text="Progreso:")
progress_label.pack(pady=5)
progress = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress.pack(pady=5)

# Función para mostrar resultados en la tabla
def display_result(results):
    # Limpiar la tabla antes de insertar nuevos resultados
    for row in results_table.get_children():
        results_table.delete(row)
    
    # Insertar los resultados en la tabla
    for subject, status in results:
        results_table.insert("", "end", values=(subject, status))

root.mainloop()