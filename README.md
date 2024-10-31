Detector de Spam
Este proyecto es un detector de spam desarrollado en Python, que utiliza Tkinter para la interfaz gráfica de usuario y scikit-learn para entrenar un modelo de detección de spam. El modelo clasifica correos electrónicos como "Spam" o "No es Spam" usando algoritmos de aprendizaje automático.

Estructura del Proyecto
La estructura de carpetas del proyecto es la siguiente:

plaintext
Copiar código
DetectorSpam/
├── data/
│   ├── Spam/
│   │   ├── spam/
│   │   └── spam_2/
│   └── ham/
│       └── hard_ham/
├── models/
│   ├── vectorizer.pkl
│   └── clf_model.pkl
├── finalAplication.py
├── README.md
├── requirements.txt
└── .gitignore
data/: Carpeta que contiene los datos de entrenamiento.
Spam/: Carpeta para los correos clasificados como spam.
spam/ y spam_2/: Subcarpetas para los correos de spam.
ham/: Carpeta para los correos clasificados como "ham" (no spam).
hard_ham/: Subcarpeta para los correos no spam.
models/: Carpeta donde se almacenarán los archivos del modelo entrenado.
vectorizer.pkl y clf_model.pkl: Archivos generados al entrenar el modelo. Estos archivos permiten clasificar correos sin necesidad de reentrenar cada vez.
finalAplication.py: Código principal del detector de spam y la interfaz de usuario.
README.md: Documentación del proyecto.
requirements.txt: Lista de dependencias necesarias para ejecutar el código.
.gitignore: Archivo que ignora ciertos archivos y carpetas al subir al repositorio.
Instalación
Clona el repositorio:

bash
Copiar código
git clone https://github.com/IsaacSs-SS/DetectorSpam.git
cd DetectorSpam
Instala las dependencias:

bash
Copiar código
pip install -r requirements.txt
Descarga los datos necesarios de nltk:

python
Copiar código
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Uso
Ejecuta main.py para iniciar el programa:

bash
Copiar código
python main.py
Usa la interfaz para:

Entrenar el modelo: Carga los datos de las carpetas data/Spam y data/ham y entrena el modelo de detección de spam.
Agregar más datos: Añade más correos electrónicos a las carpetas de datos (Spam o ham) para mejorar el entrenamiento del modelo.
Clasificar un correo individual: Escribe o pega el contenido de un correo en el campo de texto y clasifícalo como "Spam" o "No es Spam".
Clasificar todos los correos de la bandeja de entrada: Conéctate a tu cuenta de Gmail (o cualquier otra cuenta compatible con IMAP) y clasifica automáticamente todos los correos de la bandeja de entrada.
Detalles Técnicos
Entrenamiento: La función train_model() entrena el modelo utilizando TfidfVectorizer para vectorizar el texto y un Perceptron de scikit-learn como clasificador.
Clasificación:
classify_single_email(): Clasifica un correo individual ingresado manualmente.
classify_all_emails(): Clasifica todos los correos en la bandeja de entrada usando el protocolo IMAP para conectarse a una cuenta de correo.
Interfaz Gráfica: La interfaz de usuario está creada con Tkinter, permitiendo una interacción fácil y accesible con el programa.
Contribuciones
Las contribuciones son bienvenidas. Si encuentras errores o tienes sugerencias de mejora, no dudes en crear un issue o enviar un pull request.

Licencia
Este proyecto está bajo la licencia MIT.

