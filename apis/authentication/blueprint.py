import datetime
import os
from flask import Blueprint,request,jsonify,current_app
import jwt
import bcrypt
user = Blueprint('user',__name__,url_prefix='/users')

@user.route('/register',methods=['POST'])
def register():
    try:
        user = request.json
        db =  current_app.config['Mongo_db']

        user_exists = db.users.find_one({'email': user['email']})
        if user_exists:
            return jsonify({'error': 'User already exists'}), 400
        salt = bcrypt.gensalt()
        
        hashed_password = bcrypt.hashpw(user['password'].encode('utf-8'), salt)
        tenant = user.get('tenant', 'fyp_group_5')
        user_role = user.get('user_role', 1)
        username = user.get('username')
        user['password'] = hashed_password
        user['tenant'] = tenant
        user['user_role'] = user_role
        user['username'] = username
        result = db.users.insert_one(user)
        
        if result.inserted_id:
            del user['password']
            user['_id'] = str(result.inserted_id)
            payload = {
                'tenant': tenant,
                'user_role': user_role,
                'id': str(result.inserted_id),
                'email': user['email'],
                'exp': datetime.datetime.now()+datetime.timedelta(days=1)
            }
            key = os.getenv('SECRET_KEY')
            token = jwt.encode(payload, key, algorithm='HS256')
            user['token'] = token
            return jsonify({'user':user,'message': 'User created successfully'}), 201
        else:
            return jsonify({'error':"Error"}),500
    except (ValueError, TypeError) as ve_te:
        return jsonify({'error': 'Invalid data format: ' + str(ve_te)}), 400
    except Exception as e:
        print(e)       
        return jsonify({'error': 'An unexpected error occurred'}), 500

@user.route('/login',methods=['POST'])
def login():
    try:
        user = request.json
        db =  current_app.config['Mongo_db']
        user_exists = db.users.find_one({'email': user['email']})
        if user_exists:
            if bcrypt.checkpw(user['password'].encode('utf-8'), user_exists['password']):
                del user_exists['password']
                payload = {
                    'tenant': user_exists['tenant'],
                    'user_role': user_exists['user_role'],
                    'id': str(user_exists['_id']),
                    'email': user_exists['email'],
                    'exp': datetime.datetime.now()+datetime.timedelta(days=1)
                }
                key = os.getenv('SECRET_KEY')
                token = jwt.encode(payload, key, algorithm='HS256')
                user_exists['token'] = token
                user_exists['_id'] = str(user_exists['_id'])
                return jsonify({'user':user_exists,'message': 'User logged in successfully'}), 200
            else:
                return jsonify({'error': 'Invalid credentials'}), 400
    except (ValueError, TypeError) as ve_te:
        return jsonify({'error': 'Invalid data format: ' + str(ve_te)}), 400
    except Exception as e:
        print(e)
        return jsonify({'error': 'An unexpected error occurred'}), 500