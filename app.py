from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "disease_model.pkl")
SYMPTOM_PATH = os.path.join(BASE_DIR, "symptom_columns.pkl")

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  #password insert

@app.context_processor
def inject_notifications():
    if 'user' in session:
        user = session['user']
        active_alerts = Emergency_Alert.query.filter_by(status='active').count()
        unread_feedback = Feedback.query.filter_by(is_read=False).count() if user['role'] == 'admin' else 0
        notification_count = unread_feedback + active_alerts if user['role'] == 'admin' else active_alerts if user['role'] in ['doctor', 'nurse'] else 0
        return {'notification_count': notification_count, 'active_alerts': active_alerts, 'unread_feedback': unread_feedback}
    return {}

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://flask_user:12345@localhost/medicare_hospital'  #change flask_user:(password) if tukar password
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load ML model
try:
    model = joblib.load(MODEL_PATH)  # load model to use for implementation predictions
    symptom_cols = joblib.load(SYMPTOM_PATH)  # Load symptom column
    print("Final prediction model loaded successfully.")
except Exception as e:
    model = None
    symptom_cols = []
    print(f"Error loading final ML model: {e}")

symptom_mapping = {
    # General respiratory symptoms
    'fever': 'fever',
    'cough': 'cough',
    'fatigue': 'fatigue',
    'shortness of breath': 'shortness_of_breath',
    'chest pain': 'chest_pain',
    'sore throat': 'sore_throat',

    # Cold / URTI
    'runny nose': 'runny_nose',
    'sneezing': 'sneezing',

    # Asthma
    'wheezing': 'wheezing',

    # Pneumonia / Bronchitis
    'phlegm': 'phlegm',
    'difficulty breathing': 'difficulty_breathing',

    # Tuberculosis (screening only)
    'weight loss': 'weight_loss',
    'night sweats': 'night_sweats'
}

#database
class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # e.g., 'patient', 'doctor', 'admin'
    contact = db.Column(db.String(20), nullable=True)
    status = db.Column(db.String(20), default='active')  # This line is crucial

class Department(db.Model):
    __tablename__ = 'department'
    department_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    floor = db.Column(db.String(10), nullable=False)

class Appointment(db.Model):
    __tablename__ = 'appointment'
    appointment_id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    department_id = db.Column(db.Integer, db.ForeignKey('department.department_id'), nullable=False)
    appointment_datetime = db.Column(db.DateTime, nullable=False)
    type = db.Column(db.String(20), nullable=False)  # regular, follow-up, symptoms
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected

class Symptom_Entry(db.Model):
    __tablename__ = 'symptom_entry'
    symptom_id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    submitted_date = db.Column(db.Date, default=datetime.utcnow().date)
    symptoms = db.Column(db.Text, nullable=False)

class AI_Prediction(db.Model):
    __tablename__ = 'ai_prediction'
    prediction_id = db.Column(db.Integer, primary_key=True)
    symptom_id = db.Column(db.Integer, db.ForeignKey('symptom_entry.symptom_id'), nullable=False)
    predicted_disease = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)

class Emergency_Alert(db.Model):
    __tablename__ = 'emergency_alert'
    alert_id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    triggered_at = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(20), default='active')  # active, resolved
    description = db.Column(db.Text)

class Prescription(db.Model):
    __tablename__ = 'prescription'
    prescription_id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointment.appointment_id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    issued_date = db.Column(db.Date, default=datetime.utcnow().date)
    diagnosis = db.Column(db.Text, nullable=False)
    medications = db.Column(db.Text, nullable=False)

class Feedback(db.Model):
    feedback_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    submitted_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)  # For notifications
    resolved = db.Column(db.Boolean, default=False)  # For resolution

class Education_Content(db.Model):
    __tablename__ = 'education_content'
    content_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(50), nullable=False)  # 'article', 'tip', 'image'
    content_body = db.Column(db.Text, nullable=True)  # For text content
    file_path = db.Column(db.String(200), nullable=True)  # For uploaded files (images/PDFs)
    published_date = db.Column(db.Date, default=datetime.utcnow().date)

class Queue(db.Model):
    __tablename__ = 'queue'
    queue_id = db.Column(db.Integer, primary_key=True)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointment.appointment_id'), nullable=False)
    number = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(20), default='waiting')  # waiting, called, completed

class Doctor_Schedule(db.Model):
    __tablename__ = 'doctor_schedule'
    schedule_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)

class Analytics_Settings(db.Model):
    __tablename__ = 'analytics_settings'
    id = db.Column(db.Integer, primary_key=True)
    embed_url = db.Column(db.String(500), nullable=True)
    uploaded_file = db.Column(db.String(200), nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables and seed initial data
with app.app_context():
    db.create_all()
    print("DB tables created.")

    # Add file_path column if it doesn't exist
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE education_content ADD COLUMN IF NOT EXISTS file_path VARCHAR(200);"))
            conn.commit()
        print("Added file_path column to education_content table.")
    except Exception as e:
        print(f"Column already exists or error: {e}")

    # Increase title column length if necessary
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE education_content ALTER COLUMN title TYPE VARCHAR(500);"))
            conn.commit()
        print("Increased title column length to 500 characters.")
    except Exception as e:
        print(f"Error altering title column: {e}")

    # Add resolved column to feedback table if it doesn't exist
    try:
        with db.engine.connect() as conn:
            conn.execute(db.text("ALTER TABLE feedback ADD COLUMN IF NOT EXISTS resolved BOOLEAN DEFAULT FALSE;"))
            conn.commit()
        print("Added resolved column to feedback table.")
    except Exception as e:
        print(f"Column already exists or error: {e}")
    
    if not Department.query.first():
        dept1 = Department(name='Cardiology', floor='Floor 1')
        dept2 = Department(name='General Medicine', floor='Floor 2')
        dept3 = Department(name='Neurology', floor='Floor 2')
        dept4 = Department(name='Pediatrics', floor='Floor 3')
        dept5 = Department(name='Orthopedics', floor='Floor 3')
        db.session.add_all([dept1, dept2])
    
    if not User.query.filter_by(email='patient@example.com').first():
        patient = User(
            name='John Doe',
            email='patient@example.com',
            password=generate_password_hash('pass123'),
            role='patient',
            contact='1234567890'
        )
        db.session.add(patient)
    
    if not User.query.filter_by(email='doctor@example.com').first():
        doctor = User(
            name='Dr. Sarah Johnson',
            email='doctor@example.com',
            password=generate_password_hash('pass123'),
            role='doctor',
            contact='0987654321'
        )
        db.session.add(doctor)
    
    if not User.query.filter_by(email='adaniesah@gmail.com').first():
        admin = User(
            name='Admin User',
            email='adaniesah@gmail.com',
            password=generate_password_hash('qwerty'),
            role='admin',
            contact='1122334455'
        )
        db.session.add(admin)
    
    if not Doctor_Schedule.query.first():
        doctor_id = User.query.filter_by(email='doctor@example.com').first().user_id
        schedule1 = Doctor_Schedule(
            user_id=doctor_id,
            date=datetime.now().date(),
            start_time=datetime.strptime('09:00', '%H:%M').time(),
            end_time=datetime.strptime('17:00', '%H:%M').time()
        )
        db.session.add(schedule1)
    
    db.session.commit()
    print("Initial data seeded.")

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('homepage'))
    return render_template('index.html')  

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        # Query by email only (role is determined from DB)
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user'] = {'id': user.user_id, 'name': user.name, 'email': user.email, 'role': user.role}
            return redirect(url_for('homepage'))
        else:
            flash('Invalid credentials.', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        phone = request.form['phone']
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return redirect(url_for('register'))
        new_patient = User(name=name, email=email, password=password, role='patient', contact=phone)
        db.session.add(new_patient)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

@app.route('/homepage')
def homepage():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    
    if user['role'] == 'patient':
        # Fetch patient's appointments with related doctor and department data
        appointments = db.session.query(Appointment, User, Department).join(
            User, Appointment.doctor_id == User.user_id
        ).join(
            Department, Appointment.department_id == Department.department_id
        ).filter(Appointment.patient_id == user['id']).order_by(Appointment.appointment_datetime).all()
        
        # Calculate upcoming appointments (e.g., future dates)
        upcoming_count = sum(1 for appt, _, _ in appointments if appt.appointment_datetime > datetime.now())
        
        # Fetch queue positions
        queue_positions = {}
        for appt, _, _ in appointments:
            queue_entry = Queue.query.filter_by(appointment_id=appt.appointment_id).first()
            queue_positions[appt.appointment_id] = queue_entry.number if queue_entry else 'N/A'

        # Calculate people ahead in queue for the next appointment
        next_appointment = min((appt for appt, _, _ in appointments if appt.appointment_datetime > datetime.now()), key=lambda x: x.appointment_datetime, default=None)
        people_ahead = 0
        if next_appointment:
            queue_entry = Queue.query.filter_by(appointment_id=next_appointment.appointment_id).first()
            if queue_entry:
                people_ahead = Queue.query.join(Appointment).filter(
                    Appointment.doctor_id == next_appointment.doctor_id,
                    Queue.number < queue_entry.number,
                    Queue.status.in_(['waiting', 'called'])
                ).count()

        # Calculate medical records count (prescriptions as medical documents)
        medical_records_count = Prescription.query.join(Appointment).filter(Appointment.patient_id == user['id']).count()

        return render_template('homepage.html', user=user, appointments=appointments, upcoming_count=upcoming_count, queue_positions=queue_positions, people_ahead=people_ahead, medical_records_count=medical_records_count)
    elif user['role'] in ['doctor', 'nurse']:
        # Fetch today's date
        today = datetime.now().date()
        
        # Today's patients: Appointments for today assigned to this doctor
        todays_appointments = Appointment.query.filter(
            Appointment.doctor_id == user['id'],
            Appointment.appointment_datetime >= datetime.combine(today, datetime.min.time()),
            Appointment.appointment_datetime < datetime.combine(today + timedelta(days=1), datetime.min.time())
        ).all()
        todays_patients = len(todays_appointments)
        
        # In Queue: Pending appointments for today
        in_queue = len([appt for appt in todays_appointments if appt.status == 'pending'])
        
        # Completed: Approved appointments for today
        completed = len([appt for appt in todays_appointments if appt.status == 'approved'])
        
        # Pending Reviews: Placeholder (e.g., feedback or something; adjust as needed)
        pending_reviews = 3  # Replace with real logic if available
        
        # Current Queue: Pending appointments with patient details, symptoms, AI prediction
        queue_data = []
        for appt in todays_appointments:
            if appt.status == 'pending':
                patient = User.query.get(appt.patient_id)
                symptom_entry = Symptom_Entry.query.filter_by(patient_id=appt.patient_id).order_by(Symptom_Entry.submitted_date.desc()).first()
                ai_prediction = AI_Prediction.query.filter_by(symptom_id=symptom_entry.symptom_id).first() if symptom_entry else None
                
                queue_data.append({
                    'appointment_id': appt.appointment_id,
                    'patient_name': patient.name if patient else 'Unknown',
                    'patient_age': 45,  # Placeholder; add age to User model if needed
                    'chief_complaint': symptom_entry.symptoms if symptom_entry else 'N/A',
                    'ai_prediction': f"{ai_prediction.predicted_disease} ({ai_prediction.confidence_score:.0%} confidence)" if ai_prediction else 'N/A'
                })
        
        return render_template('homepage.html', user=user, todays_patients=todays_patients, in_queue=in_queue, completed=completed, pending_reviews=pending_reviews, queue_data=queue_data)
    else:
        # For other roles (e.g., admin)
        return render_template('homepage.html', user=user)

# New route for Start Consultation button
@app.route('/start_consultation/<int:appointment_id>', methods=['POST'])
def start_consultation(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id'], status='approved').first()
    if appointment:
        return redirect(url_for('consultation', appointment_id=appointment_id))  # Redirect to consultation page
    else:
        flash('Appointment not found or not authorized.', 'error')
        return redirect(url_for('homepage'))

@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    active_alerts = Emergency_Alert.query.filter_by(status='active').count()
    unread_feedback = Feedback.query.filter_by(is_read=False).count() if user['role'] == 'admin' else 0
    notification_count = unread_feedback + active_alerts if user['role'] == 'admin' else active_alerts if user['role'] in ['doctor', 'nurse'] else 0
    prediction = None
    confidence = None
    
    if request.method == 'POST':
        if model is None:
            flash('AI model not loaded. Please contact admin.', 'error')
        else:
            try:
                # Collect symptoms from form
                symptoms_list = request.form.getlist('symptoms')  # Checkboxes
                other_symptoms = request.form.get('other-symptoms', '').lower().split(',')  # Text area
                all_symptoms = [s.strip().lower() for s in symptoms_list + other_symptoms if s.strip()]
                
                # Mapping dict
                mapped_symptoms = [symptom_mapping.get(s, s) for s in all_symptoms]  # Fallback to original if no mapping
                print(f"Debug: Original symptoms: {all_symptoms}")  # Debug
                print(f"Debug: Mapped symptoms: {mapped_symptoms}")  # Debug
                print(f"Debug: Model expects: {len(symptom_cols)} columns, e.g., {symptom_cols[:5]}")  # Debug
                
                # Ensure length matches symptom_cols
                features = [1 if col in mapped_symptoms else 0 for col in symptom_cols]
                print(f"Debug: Feature vector length: {len(features)} (should be {len(symptom_cols)})")  # Debug
                
                if len(features) != len(symptom_cols):
                    raise ValueError(f"Feature vector length mismatch: {len(features)} vs {len(symptom_cols)}. Check symptom mapping.")
                
                # Prediction
                prediction = model.predict([features])[0]
                confidence = model.predict_proba([features])[0].max()
                print(f"Debug: Prediction: {prediction}, Confidence: {confidence}")  # Debug
                
                # Save to database
                symptom_entry = Symptom_Entry(patient_id=user['id'], symptoms=', '.join(all_symptoms))
                db.session.add(symptom_entry)
                db.session.commit()
                
                ai_prediction = AI_Prediction(
                    symptom_id=symptom_entry.symptom_id,
                    predicted_disease=prediction,
                    confidence_score=float(confidence)  
                )
                db.session.add(ai_prediction)
                db.session.commit()
                
                flash(f'AI Prediction: {prediction} (Confidence: {confidence:.2f})', 'info')
            except Exception as e:
                print(f"Error during prediction: {e}")  # Full error in terminal
                flash(f'An error occurred: {str(e)}. Check symptoms and try again.', 'error')
                prediction = None
                confidence = None
    
    # Fetch recent predictions for display
    recent_predictions = AI_Prediction.query.join(Symptom_Entry).filter(Symptom_Entry.patient_id == user['id']).order_by(AI_Prediction.prediction_id.desc()).limit(5).all()
    
    return render_template('symptoms-view.html', user=user, prediction=prediction, confidence=confidence, recent_predictions=recent_predictions)

@app.route('/navigation')
def navigation():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    return render_template('navigation-view.html', user=user)

@app.route('/book-appointment', methods=['GET', 'POST'])
def book_appointment():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if request.method == 'POST':
        selected_date = request.form['date']
        selected_time = request.form['time']
        appointment_datetime = datetime.strptime(selected_date + ' ' + selected_time, '%Y-%m-%d %H:%M')
        
        # Find available doctors for the selected date
        available_doctors = []
        doctor_schedules = Doctor_Schedule.query.filter_by(date=appointment_datetime.date()).all()
        for schedule in doctor_schedules:
            doctor = User.query.get(schedule.user_id)
            if doctor and doctor.role == 'doctor':
                # Check if the doctor has an appointment at this time
                existing_appointment = Appointment.query.filter(
                    Appointment.doctor_id == doctor.user_id,
                    Appointment.appointment_datetime == appointment_datetime
                ).first()
                if not existing_appointment:
                    available_doctors.append(doctor)
        
        if not available_doctors:
            flash('No doctors available at the selected time. Please choose another time.', 'error')
            return redirect(url_for('book_appointment'))
        
        # Assign to the first available doctor
        doctor_id = available_doctors[0].user_id
        
        # Assign to default department (General Medicine) since department selection is removed
        default_dept = Department.query.filter_by(name='General Medicine').first()
        department_id = default_dept.department_id if default_dept else 1  # Fallback to 1

        appointment = Appointment(
            patient_id=user['id'],
            doctor_id=doctor_id,
            department_id=department_id,
            appointment_datetime=appointment_datetime,
            type=request.form['type']
        )
        db.session.add(appointment)

        # Assign queue number
        # Get the max queue number for this doctor's appointments
        max_queue_number = db.session.query(db.func.max(Queue.number)).join(Appointment).filter(Appointment.doctor_id == doctor_id).scalar() or 0
        queue_entry = Queue(appointment_id=appointment.appointment_id, number=max_queue_number + 1, status='waiting')
        db.session.add(queue_entry)

        db.session.commit()

        # Create notification for admin and doctors
        alert = Emergency_Alert(
            patient_id=user['id'],
            description=f"New appointment booked by {user['name']} for {appointment_datetime.strftime('%Y-%m-%d %H:%M')} with Dr. {doctor.name}"
        )
        db.session.add(alert)
        db.session.commit()

        flash('Appointment submitted successfully!', 'success')
        return redirect(url_for('appointment_history'))
    
    # GET request: Prepare data
    symptoms = request.args.get('symptoms', '')
    departments = Department.query.all()
    
    # Get selected date from query params or default to today
    selected_date_str = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    selected_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
    
    # Fetch available doctors for the selected date
    available_doctors = []
    doctor_schedules = Doctor_Schedule.query.filter_by(date=selected_date).all()
    for schedule in doctor_schedules:
        doctor = User.query.get(schedule.user_id)
        if doctor and doctor.role == 'doctor':
            available_doctors.append({
                'id': doctor.user_id,
                'name': doctor.name,
                'start_time': schedule.start_time,
                'end_time': schedule.end_time
            })
    
    # Suggest soonest available time for the first available doctor (if any)
    suggested_time = None
    if available_doctors:
        first_doctor = available_doctors[0]
        start_time = datetime.combine(selected_date, first_doctor['start_time'])
        end_time = datetime.combine(selected_date, first_doctor['end_time'])
        
        # Find booked times for this doctor on this date
        booked_times = [
            appt.appointment_datetime.time()
            for appt in Appointment.query.filter(
                Appointment.doctor_id == first_doctor['id'],
                Appointment.appointment_datetime >= start_time,
                Appointment.appointment_datetime < end_time
            ).all()
        ]
        
        # Suggest next 30-min slot
        current_time = start_time
        while current_time < end_time:
            if current_time.time() not in booked_times:
                suggested_time = current_time.time().strftime('%H:%M')
                break
            current_time += timedelta(minutes=30)
    
    return render_template('book-appointment.html', 
                           symptoms=symptoms, 
                           user=user, 
                           departments=departments, 
                           available_doctors=available_doctors, 
                           selected_date=selected_date_str, 
                           suggested_time=suggested_time)

@app.route('/appointment-history')
def appointment_history():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    
    # Fetch appointments with related user data
    if user['role'] == 'doctor':
        appointments = db.session.query(Appointment, User).join(User, Appointment.patient_id == User.user_id).filter(Appointment.doctor_id == user['id']).all()
    else:
        appointments = db.session.query(Appointment, User).join(User, Appointment.doctor_id == User.user_id).filter(Appointment.patient_id == user['id']).all()
    
    # Enrich data for template
    enriched_appointments = []
    for appt, related_user in appointments:
        enriched_appointments.append({
            'date': appt.appointment_datetime.strftime('%Y-%m-%d'),
            'time': appt.appointment_datetime.strftime('%H:%M'),
            'related_name': related_user.name,  # Doctor or patient name
            'reason': appt.type,  # Using 'type' as reason
            'status': appt.status
        })
    
    message = session.pop('appointment_booked', None)
    return render_template('appointment-history.html', appointments=enriched_appointments, message=message, user=user)

@app.route('/records')
def records():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if user['role'] == 'patient':
        # Fetch patient's data from DB
        appointments = Appointment.query.filter_by(patient_id=user['id']).all()
        prescriptions = Prescription.query.join(Appointment).filter(Appointment.patient_id == user['id']).all()
        symptom_entries = Symptom_Entry.query.filter_by(patient_id=user['id']).all()
        ai_predictions = AI_Prediction.query.join(Symptom_Entry).filter(Symptom_Entry.patient_id == user['id']).all()
        emergency_alerts = Emergency_Alert.query.filter_by(patient_id=user['id']).all()
        
        # Pass data to template
        return render_template('records-view.html', 
                               user=user, 
                               appointments=appointments,
                               prescriptions=prescriptions, 
                               symptom_entries=symptom_entries, 
                               ai_predictions=ai_predictions, 
                               emergency_alerts=emergency_alerts)
    else:
        # For doctors/nurses: placeholder
        return render_template('records-view.html', user=user)

@app.route('/emergency')
def emergency():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if user['role'] == 'admin':
        active_alerts = db.session.query(Emergency_Alert, User.name.label('patient_name')).join(User, Emergency_Alert.patient_id == User.user_id).filter(Emergency_Alert.status == 'active').all()
        active_alerts = [{'alert_id': alert.alert_id, 'patient_name': name, 'description': alert.description, 'triggered_at': alert.triggered_at} for alert, name in active_alerts]
    else:
        active_alerts = []
    return render_template('emergency-view.html', user=user, active_alerts=active_alerts)

@app.route('/trigger-alert', methods=['POST'])
def trigger_alert():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    alert = Emergency_Alert(
        patient_id=user['id'],
        description=request.form.get('description', '')
    )
    db.session.add(alert)
    db.session.commit()
    flash('Emergency alert triggered!', 'success')
    return redirect(url_for('emergency'))

@app.route('/queue')
def queue():
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']

    # Fetch queue entries with appointments for the logged-in doctor, ordered by queue number
    queue_entries = db.session.query(Queue, Appointment, User).join(
        Appointment, Queue.appointment_id == Appointment.appointment_id
    ).join(
        User, Appointment.patient_id == User.user_id
    ).filter(
        Appointment.doctor_id == user['id'],
        Appointment.status.in_(['pending', 'approved'])  # Include both
    ).order_by(Queue.number).all()

    # Format for template
    queue = []
    for q_entry, appt, patient in queue_entries:
        queue.append({
            'appointment_id': appt.appointment_id,
            'position': q_entry.number,
            'patient': patient.name if patient else 'Unknown',
            'estimated_wait': '10-30 min',  # Mock
            'reason': appt.type,
            'status': appt.status
        })

    return render_template('queue-view.html', queue=queue, user=user)

# New route for starting consultation from queue
@app.route('/start_consultation_queue/<int:appointment_id>', methods=['POST'])
def start_consultation_queue(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id'], status='approved').first()
    if appointment:
        appointment.status = 'in_progress'  # Or 'completed'; adjust as needed
        db.session.commit()
        flash('Consultation started!', 'success')
    else:
        flash('Appointment not found or not authorized.', 'error')
    
    return redirect(url_for('queue'))

@app.route('/approve_appointment/<int:appointment_id>', methods=['POST'])
def approve_appointment(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    

    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id'], status='pending').first()
    if appointment:
        appointment.status = 'approved'
        db.session.commit()
        flash('Appointment approved successfully!', 'success')
    else:
        flash('Appointment not found or not authorized.', 'error')
    
    return redirect(url_for('queue'))

@app.route('/reject_appointment/<int:appointment_id>', methods=['POST'])
def reject_appointment(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    # Find and update the appointment
    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id'], status='pending').first()
    if appointment:
        appointment.status = 'rejected'
        # Update queue status
        queue_entry = Queue.query.filter_by(appointment_id=appointment_id).first()
        if queue_entry:
            queue_entry.status = 'cancelled'
        db.session.commit()
        flash('Appointment rejected.', 'error')  # Use 'error' for rejection
    else:
        flash('Appointment not found or not authorized.', 'error')
    
    return redirect(url_for('queue'))

@app.route('/schedule')
def schedule():
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse', 'admin']:
        return redirect(url_for('login'))
    user = session['user']
    doctor_id = request.args.get('doctor_id')
    if doctor_id and user['role'] == 'admin':
        # Load specific doctor's schedule
        my_schedule = Doctor_Schedule.query.filter_by(user_id=doctor_id).all()  # Use Doctor_Schedule model
        time_off_requests = []  # Placeholder: Schedule_Request model not defined
        all_schedules = []  # Admin can see all, but for specific, filter
    else:
        # Default for doctor/nurse
        my_schedule = Doctor_Schedule.query.filter_by(user_id=user['id']).all()
        time_off_requests = []  # Placeholder: Schedule_Request model not defined
        all_schedules = []
    doctors = User.query.filter(User.role.in_(['doctor', 'nurse'])).all()

    # For admin, populate all_schedules with all doctor schedules including doctor names
    if user['role'] == 'admin':
        all_schedules = db.session.query(Doctor_Schedule, User.name.label('doctor_name')).join(
            User, Doctor_Schedule.user_id == User.user_id
        ).all()
        all_schedules = [{'id': sched.schedule_id, 'doctor_name': name, 'date': sched.date.strftime('%Y-%m-%d'), 'start_time': sched.start_time.strftime('%H:%M'), 'end_time': sched.end_time.strftime('%H:%M')} for sched, name in all_schedules]

    return render_template('schedule-view.html', user=user, my_schedule=my_schedule, time_off_requests=time_off_requests, all_schedules=all_schedules, doctors=doctors)

@app.route('/assign-schedule', methods=['POST'])
def assign_schedule():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    
    doctor_id = request.form.get('doctor_id')
    date = request.form.get('date')
    start_time = request.form.get('start_time')
    end_time = request.form.get('end_time')
    
    if not doctor_id or not date or not start_time or not end_time:
        flash('All fields are required.', 'error')
        return redirect(url_for('schedule'))
    
    try:
        new_schedule = Doctor_Schedule(
            user_id=int(doctor_id),
            date=datetime.strptime(date, '%Y-%m-%d').date(),
            start_time=datetime.strptime(start_time, '%H:%M').time(),
            end_time=datetime.strptime(end_time, '%H:%M').time()
        )
        db.session.add(new_schedule)
        db.session.commit()
        flash('Schedule assigned successfully!', 'success')
    except Exception as e:
        flash(f'Error assigning schedule: {str(e)}', 'error')
    
    return redirect(url_for('schedule'))

@app.route('/add_schedule', methods=['POST'])
def add_schedule():
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    # Get form data
    date = request.form['date']
    start_time = request.form['start_time']
    end_time = request.form['end_time']
    
    # Validate and add
    try:
        new_schedule = Doctor_Schedule(
            user_id=user['id'],
            date=datetime.strptime(date, '%Y-%m-%d').date(),
            start_time=datetime.strptime(start_time, '%H:%M').time(),
            end_time=datetime.strptime(end_time, '%H:%M').time()
        )
        db.session.add(new_schedule)
        db.session.commit()
        flash('Schedule added successfully!', 'success')
    except Exception as e:
        flash(f'Error adding schedule: {str(e)}', 'error')
    
    return redirect(url_for('schedule'))

@app.route('/edit_schedule/<int:schedule_id>', methods=['GET', 'POST'])
def edit_schedule(schedule_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    # Find the schedule (only if owned by user)
    sched = Doctor_Schedule.query.filter_by(schedule_id=schedule_id, user_id=user['id']).first()
    if not sched:
        flash('Schedule not found or not authorized.', 'error')
        return redirect(url_for('schedule'))
    
    if request.method == 'POST':
        # Update with form data
        sched.date = datetime.strptime(request.form['date'], '%Y-%m-%d').date()
        sched.start_time = datetime.strptime(request.form['start_time'], '%H:%M').time()
        sched.end_time = datetime.strptime(request.form['end_time'], '%H:%M').time()
        db.session.commit()
        flash('Schedule updated successfully!', 'success')
        return redirect(url_for('schedule'))
    
    # GET: Show edit form
    return render_template('edit-schedule.html', schedule=sched, user=user)  # Create this template if needed

@app.route('/delete_schedule/<int:schedule_id>', methods=['POST'])
def delete_schedule(schedule_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    
    # Find and delete
    sched = Doctor_Schedule.query.filter_by(schedule_id=schedule_id, user_id=user['id']).first()
    if sched:
        db.session.delete(sched)
        db.session.commit()
        flash('Schedule deleted successfully!', 'success')
    else:
        flash('Schedule not found or not authorized.', 'error')
    
    return redirect(url_for('schedule'))

@app.route('/patient-records')
def patient_records():
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse', 'admin']:
        return redirect(url_for('login'))
    user = session['user']
    patients = User.query.filter_by(role='patient').all()
    patient_id = request.args.get('patient_id')
    if patient_id and user['role'] == 'admin':
        # Load specific patient's data for admin
        patient_history = []  # Placeholder: Consultation model not defined
        patient_prescriptions = Prescription.query.filter_by(patient_id=patient_id).all()
        patient_symptoms = Symptom_Entry.query.filter_by(patient_id=patient_id).all()  # Use Symptom_Entry instead
    else:
        # Default empty for doctor/nurse
        patient_history = []
        patient_prescriptions = []
        patient_symptoms = []
@app.route('/feedback')
def feedback():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    if user['role'] == 'admin':
        notifications = Feedback.query.filter_by(is_read=False).all()
        all_feedback = Feedback.query.join(User, Feedback.user_id == User.user_id).add_columns(User.name, User.role).all()
        return render_template('feedback-view.html', user=user, notifications=notifications, all_feedback=all_feedback)
    else:
        notifications = []
        all_feedback = []
        return render_template('feedback-view.html', user=user, notifications=notifications, all_feedback=all_feedback)

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    category = request.form['category']
    rating = int(request.form['rating'])
    comments = request.form.get('comments', '')
    
    feedback = Feedback(user_id=user['id'], category=category, rating=rating, comments=comments)
    db.session.add(feedback)
    db.session.commit()
    
    flash('Feedback submitted successfully!', 'success')
    return redirect(url_for('feedback'))

@app.route('/mark-feedback-read/<int:feedback_id>', methods=['POST'])
def mark_feedback_read(feedback_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    feedback = Feedback.query.get(feedback_id)
    if feedback:
        feedback.is_read = True
        db.session.commit()
    return jsonify({'success': True})

@app.route('/resolve-feedback/<int:feedback_id>', methods=['POST'])
def resolve_feedback(feedback_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    feedback = Feedback.query.get(feedback_id)
    if feedback:
        feedback.resolved = True
        db.session.commit()
        # Here you could add notification logic, e.g., send email to the user
        flash('Feedback resolved and user notified!', 'success')
    return jsonify({'success': True})

@app.route('/analytics')
def analytics():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    # Fetch embed_url and uploaded_file from DB
    settings = Analytics_Settings.query.first()
    embed_url = settings.embed_url if settings else None
    uploaded_file = settings.uploaded_file if settings else None
    return render_template('analytics-dashboard.html', user=user, embed_url=embed_url, uploaded_file=uploaded_file)

@app.route('/update-analytics-settings', methods=['POST'])
def update_analytics_settings():
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    embed_url = request.form.get('embed_url')
    uploaded_file = request.files.get('uploaded_file')

    settings = Analytics_Settings.query.first()
    if not settings:
        settings = Analytics_Settings()
        db.session.add(settings)

    if embed_url:
        settings.embed_url = embed_url

    if uploaded_file:
        # Save the file to a directory, e.g., static/uploads
        upload_dir = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.filename)
        uploaded_file.save(file_path)
        settings.uploaded_file = uploaded_file.filename

    settings.updated_at = datetime.utcnow()
    db.session.commit()

    flash('Analytics settings updated successfully!', 'success')
    return redirect(url_for('analytics'))

@app.route('/analytics/metrics')
def analytics_metrics():
    if 'user' not in session:
        return jsonify({'error': 'Unauthorized'}), 403

    # Calculate real-time metrics
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Symptom checker submissions
    daily_symptoms = Symptom_Entry.query.filter_by(submitted_date=today).count()
    weekly_symptoms = Symptom_Entry.query.filter(
        Symptom_Entry.submitted_date >= today - timedelta(days=7)
    ).count()

    # Most common symptoms (from recent entries)
    recent_symptoms = Symptom_Entry.query.filter(
        Symptom_Entry.submitted_date >= today - timedelta(days=7)
    ).all()
    symptom_counts = {}
    for entry in recent_symptoms:
        symptoms = [s.strip().lower() for s in entry.symptoms.split(',')]
        for symptom in symptoms:
            symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1

    top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    most_common_symptoms = ', '.join([f"{s[0]} ({s[1]})" for s in top_symptoms]) if top_symptoms else 'No data'

    # Disease distribution
    recent_predictions = AI_Prediction.query.join(Symptom_Entry).filter(
        Symptom_Entry.submitted_date >= today - timedelta(days=7)
    ).all()
    disease_counts = {}
    for pred in recent_predictions:
        disease_counts[pred.predicted_disease] = disease_counts.get(pred.predicted_disease, 0) + 1

    top_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    disease_distribution = ', '.join([f"{d[0]} ({d[1]})" for d in top_diseases]) if top_diseases else 'No data'

    # Alerts triggered
    pending_alerts = Emergency_Alert.query.filter_by(status='active').count()
    resolved_alerts = Emergency_Alert.query.filter_by(status='resolved').count()

    # Appointments today
    todays_appointments = Appointment.query.filter(
        Appointment.appointment_datetime >= datetime.combine(today, datetime.min.time()),
        Appointment.appointment_datetime < datetime.combine(today + timedelta(days=1), datetime.min.time())
    ).count()

    # User activity (logins today - approximate with symptom submissions)
    active_users_today = Symptom_Entry.query.filter_by(submitted_date=today).distinct(Symptom_Entry.patient_id).count()

    return jsonify({
        'symptom_submissions': {
            'daily': daily_symptoms,
            'weekly': weekly_symptoms
        },
        'most_common_symptoms': most_common_symptoms,
        'disease_distribution': disease_distribution,
        'alerts': {
            'pending': pending_alerts,
            'resolved': resolved_alerts
        },
        'appointments_today': todays_appointments,
        'active_users_today': active_users_today,
        'last_updated': datetime.now().strftime('%H:%M:%S')
    })

@app.route('/knowledge-base')
def knowledge_base():
    if 'user' not in session:
        return redirect(url_for('login'))
    user = session['user']
    disease = request.args.get('disease')
    disease_info = {
        'Common Cold': 'A viral infection causing runny nose, cough, and fatigue. Rest and fluids help.',
        # Add more diseases
    }.get(disease, 'Information not available.')

    # Fetch content from DB
    articles = Education_Content.query.filter_by(category='article').all()
    tips = Education_Content.query.filter_by(category='tip').all()
    images = Education_Content.query.filter_by(category='image').all()

    return render_template('knowledge-base.html', user=user, disease=disease, disease_info=disease_info, articles=articles, tips=tips, images=images)

@app.route('/add_education_content', methods=['POST'])
def add_education_content():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']

    content_type = request.form.get('content_type')
    title = request.form.get('title')
    content_body = request.form.get('content')

    file_path = None
    if 'file' in request.files:
        file = request.files['file']
        if file and file.filename:
            upload_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, file.filename)
            file.save(file_path)
            file_path = file.filename  # Store relative path

    new_content = Education_Content(
        title=title,
        category=content_type,
        content_body=content_body,
        file_path=file_path
    )
    db.session.add(new_content)
    db.session.commit()
    flash('Content added successfully!', 'success')
    return redirect(url_for('knowledge_base'))

@app.route('/delete_education_content/<int:content_id>', methods=['POST'])
def delete_education_content(content_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']

    content = Education_Content.query.get(content_id)
    if content:
        # Delete file if exists
        if content.file_path:
            file_path = os.path.join(app.root_path, 'static', 'uploads', content.file_path)
            if os.path.exists(file_path):
                os.remove(file_path)
        db.session.delete(content)
        db.session.commit()
        flash('Content deleted successfully!', 'success')
    else:
        flash('Content not found.', 'error')
    return redirect(url_for('knowledge_base'))
@app.route('/consultation', methods=['GET', 'POST'])
def consultation():
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    appointment_id = request.args.get('appointment_id')
    if request.method == 'POST':
        appointment_id = request.form.get('appointment_id')
        appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id']).first()
        if appointment:
            diagnosis = request.form.get('diagnosis')
            notes = request.form.get('notes')
            prescription = request.form.get('prescription')
            prescription_obj = Prescription(
                appointment_id=appointment_id,
                doctor_id=user['id'],
                diagnosis=diagnosis,
                medications=prescription
            )
            db.session.add(prescription_obj)
            appointment.status = 'completed'
            # Update queue status
            queue_entry = Queue.query.filter_by(appointment_id=appointment_id).first()
            if queue_entry:
                queue_entry.status = 'completed'
            db.session.commit()
            flash('Consultation saved successfully!', 'success')
            return redirect(url_for('queue'))
        else:
            flash('Appointment not found.', 'error')
    if appointment_id:
        appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id']).first()
        patient = User.query.get(appointment.patient_id) if appointment else None
        symptom_entry = Symptom_Entry.query.filter_by(patient_id=appointment.patient_id).order_by(Symptom_Entry.submitted_date.desc()).first() if appointment else None
        ai_prediction = AI_Prediction.query.filter_by(symptom_id=symptom_entry.symptom_id).first() if symptom_entry else None
        return render_template('consultation-view.html', user=user, appointment=appointment, patient=patient, symptom_entry=symptom_entry, ai_prediction=ai_prediction)
    return render_template('consultation-view.html', user=user)

@app.route('/start_consultation_page/<int:appointment_id>', methods=['GET', 'POST'])
def start_consultation_page(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id']).first()
    if not appointment:
        flash('Appointment not found or not authorized.', 'error')
        return redirect(url_for('queue'))
    if request.method == 'POST':
        diagnosis = request.form.get('diagnosis', '').strip()
        notes = request.form.get('notes', '').strip()
        prescription = request.form.get('prescription', '').strip()

        if not diagnosis:
            flash('Diagnosis is required.', 'error')
            return redirect(url_for('start_consultation_page', appointment_id=appointment_id))

        if not prescription:
            flash('Prescription is required.', 'error')
            return redirect(url_for('start_consultation_page', appointment_id=appointment_id))

        prescription_obj = Prescription(
            appointment_id=appointment_id,
            doctor_id=user['id'],
            diagnosis=diagnosis,
            medications=prescription
        )
        db.session.add(prescription_obj)
        appointment.status = 'completed'
        db.session.commit()
        flash('Consultation saved successfully!', 'success')
        return redirect(url_for('queue'))
    patient = User.query.get(appointment.patient_id)
    symptom_entry = Symptom_Entry.query.filter_by(patient_id=appointment.patient_id).order_by(Symptom_Entry.submitted_date.desc()).first()
    ai_prediction = AI_Prediction.query.filter_by(symptom_id=symptom_entry.symptom_id).first() if symptom_entry else None
    return render_template('start-consultation.html', user=user, appointment=appointment, patient=patient, symptom_entry=symptom_entry, ai_prediction=ai_prediction)

@app.route('/trigger_internal_alert/<int:appointment_id>', methods=['POST'])
def trigger_internal_alert(appointment_id):
    if 'user' not in session or session['user']['role'] not in ['doctor', 'nurse']:
        return redirect(url_for('login'))
    user = session['user']
    appointment = Appointment.query.filter_by(appointment_id=appointment_id, doctor_id=user['id']).first()
    if not appointment:
        flash('Appointment not found or not authorized.', 'error')
        return redirect(url_for('start_consultation_page', appointment_id=appointment_id))
    patient = User.query.get(appointment.patient_id)
    alert = Emergency_Alert(
        patient_id=appointment.patient_id,
        description=f'Internal emergency alert triggered by Dr. {user["name"]} during consultation for patient {patient.name if patient else "Unknown"}.'
    )
    db.session.add(alert)
    db.session.commit()
    flash('Internal emergency alert triggered! Admins and other doctors have been notified.', 'success')
    return redirect(url_for('start_consultation_page', appointment_id=appointment_id))

@app.route('/resolve_alert/<int:alert_id>', methods=['POST'])
def resolve_alert(alert_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    alert = Emergency_Alert.query.get(alert_id)
    if alert:
        alert.status = 'resolved'
        db.session.commit()
        flash('Alert resolved successfully!', 'success')
    else:
        flash('Alert not found.', 'error')
    return redirect(url_for('emergency'))

@app.route('/user-management')
def user_management():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    users = User.query.all()
    return render_template('user-management.html', users=users, user=user)

@app.route('/edit-user-role', methods=['POST'])
def edit_user_role():
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    user = User.query.get(data['user_id'])
    if user:
        user.role = data['role']
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'User not found'}), 404

@app.route('/suspend-user', methods=['POST'])
def suspend_user():
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    user = User.query.get(data['user_id'])
    if user:
        user.status = 'suspended'  # Assuming you have a status field
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'User not found'}), 404

@app.route('/remove-user', methods=['POST'])
def remove_user():
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    user = User.query.get(data['user_id'])
    if user:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'User not found'}), 404

@app.route('/add-user', methods=['POST'])
def add_user():
    if 'user' not in session or session['user']['role'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    data = request.get_json()
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    new_user = User(
        name=data['name'],
        email=data['email'],
        password=generate_password_hash(data['password']),
        role=data['role'],
        contact=data.get('contact')
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True})

@app.route('/appointment-oversight')
def appointment_oversight():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    appointments = db.session.query(Appointment, User, Department).join(
        User, Appointment.patient_id == User.user_id
    ).join(
        Department, Appointment.department_id == Department.department_id
    ).all()
    enriched_appointments = []
    for appt, patient, dept in appointments:
        enriched_appointments.append({
            'appointment_id': appt.appointment_id,
            'date': appt.appointment_datetime.strftime('%Y-%m-%d'),
            'time': appt.appointment_datetime.strftime('%H:%M'),
            'patient': patient.name,
            'doctor': User.query.get(appt.doctor_id).name if User.query.get(appt.doctor_id) else 'Unknown',
            'status': appt.status
        })
    return render_template('appointment-oversight.html', appointments=enriched_appointments, user=user)

@app.route('/cancel_appointment/<int:appointment_id>', methods=['POST'])
def cancel_appointment(appointment_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    appointment = Appointment.query.get(appointment_id)
    if appointment:
        appointment.status = 'cancelled'
        db.session.commit()
        flash('Appointment cancelled successfully!', 'success')
    else:
        flash('Appointment not found.', 'error')
    return redirect(url_for('appointment_oversight'))

@app.route('/reschedule_appointment/<int:appointment_id>', methods=['POST'])
def reschedule_appointment(appointment_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    appointment = Appointment.query.get(appointment_id)
    if appointment:
        new_date = request.form.get('new_date')
        new_time = request.form.get('new_time')
        if new_date and new_time:
            new_datetime = datetime.strptime(new_date + ' ' + new_time, '%Y-%m-%d %H:%M')
            appointment.appointment_datetime = new_datetime
            db.session.commit()
            flash('Appointment rescheduled successfully!', 'success')
        else:
            flash('Invalid date or time.', 'error')
    else:
        flash('Appointment not found.', 'error')
    return redirect(url_for('appointment_oversight'))

@app.route('/ai-monitoring')
def ai_monitoring():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    raw_predictions = AI_Prediction.query.all()
    predictions = []
    for pred in raw_predictions:
        symptom_entry = Symptom_Entry.query.get(pred.symptom_id)
        patient = User.query.get(symptom_entry.patient_id) if symptom_entry else None
        predictions.append({
            'patient': patient.name if patient else 'Unknown',
            'symptoms': symptom_entry.symptoms if symptom_entry else 'N/A',
            'prediction': pred.predicted_disease,
            'confidence': f"{pred.confidence_score:.2f}",
            'accuracy': 'N/A'  # Placeholder, as accuracy isn't stored
        })
    return render_template('ai-monitoring.html', predictions=predictions, user=user)



@app.route('/system-config')
def system_config():
    if 'user' not in session or session['user']['role'] != 'admin':
        return redirect(url_for('login'))
    user = session['user']
    return render_template('system-config.html', user=user)

if __name__ == '__main__':
    app.run(debug=True)
