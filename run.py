
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate(r"C:\Users\angie\PycharmProjects\Firebase_Connection\smartmirrai-c2051-firebase-adminsdk-pnq5k-19437db28a.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://smartmirrai-c2051-default-rtdb.firebaseio.com/'})

ref = db.reference('images')
print(ref.get())

