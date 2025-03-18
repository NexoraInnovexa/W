from flask_sqlalchemy import SQLAlchemy
from app import db
from flask_login import UserMixin
from datetime import datetime
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired
from datetime import datetime
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Define Follow model first
class Follow(db.Model):
    __tablename__ = 'follows'

    id = db.Column(db.Integer, primary_key=True)
    follower_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    followed_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Relationships for follower and followed
    follower = db.relationship('User', foreign_keys=[follower_user_id])
    followed = db.relationship('User', foreign_keys=[followed_user_id])

    def __repr__(self):
        return f'<Follow {self.follower_user_id} -> {self.followed_user_id}>'

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    profile_picture = db.Column(db.String(200), nullable=True)  # Optional
    email = db.Column(db.String(120), unique=True, nullable=False)  # Fixed duplicate email field
    password = db.Column(db.String(200), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=True)  # Optional
    country = db.Column(db.String(100), nullable=True)  # Optional
    address = db.Column(db.String(200), nullable=True)  # Optional
    id_verified = db.Column(db.Boolean, default=False)
    accepted_terms = db.Column(db.Boolean, default=False)
    surname = db.Column(db.String(200), nullable=True)  # Optional
    first_name = db.Column(db.String(200), nullable=True)  # Optional
    store_name = db.Column(db.String(150), nullable=True)  # Optional
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    middle_name = db.Column(db.String(200), nullable=True)  # Optional
    role = db.Column(db.String(50), nullable=True, default="user")  # Default role
    blue_tick = db.Column(db.Boolean, default=False)
    is_storyteller = db.Column(db.Boolean, default=False)
    premium_status = db.Column(db.Boolean, default=False)
    bank_account_number = db.Column(db.String(20), nullable=True)  # Optional
    bank_name = db.Column(db.String(100), nullable=True)  # Optional
    bank_account_type = db.Column(db.String(20), nullable=True)  # Optional
    paystack_account_id = db.Column(db.String(100), nullable=True)  # Optional
    premium_expiry = db.Column(db.DateTime, nullable=True)  # Optional

    # ✅ Added relationships for followers and followed users
    followed_users = db.relationship(
        'Follow',
        foreign_keys='Follow.follower_user_id',
        backref='follower_user',
        lazy='dynamic'
    )

    followers = db.relationship(
        'Follow',
        foreign_keys='Follow.followed_user_id',
        backref='followed_user',
        lazy='dynamic'
    )

    def get_id(self):
        return str(self.id)

    def __repr__(self):
        return f'<User {self.username}>'


class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    likes = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)
    user = db.relationship('User', backref='posts')
    tags = db.relationship('Tag', secondary='post_tags', backref='posts')
    comments = db.relationship('Comment', backref='post', cascade="all, delete-orphan")
    groups_id = db.Column(db.String(36), db.ForeignKey('groups.id'), nullable=True)
    groups = db.relationship('Group', backref=db.backref('posts', lazy=True)) 

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='comments')
post_tags = db.Table('post_tags',
    db.Column('post_id', db.Integer, db.ForeignKey('post.id'), primary_key=True),
    db.Column('tag_id', db.Integer, db.ForeignKey('tag.id'), primary_key=True)
)

class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

    def __repr__(self):
        return f'<Tag {self.name}>'
    
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    visits = db.Column(db.Integer, default=0) 
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    quantity = db.Column(db.Integer, nullable=False)
    tags = db.Column(db.String(200), nullable=True)  # Tags like 'sports', 'technology', etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    images = db.Column(db.ARRAY(db.String), nullable=True) 
    def __init__(self, name, description, price, quantity, tags, images, user_id):
        self.name = name
        self.description = description
        self.price = price
        self.quantity = quantity
        self.tags = tags
        self.images = images
        self.user_id = user_id

    # Foreign key linking product to the seller (User)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    def __repr__(self):
        return f'<Product {self.name}>'
    
class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quantity = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(50), default='Pending', nullable=False)  # Pending, Shipped, Delivered, etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    total_price = db.Column(db.Float, nullable=False)

    # Relationship with DeliveryForm
    delivery_form = db.relationship('DeliveryForm', backref='order', uselist=False)

    def __repr__(self):
        return f'<Order {self.id}>'  


class DeliveryForm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    address = db.Column(db.String(500), nullable=False)
    delivery_time = db.Column(db.DateTime, nullable=False)

    # Foreign key linking DeliveryForm to Order
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False)

    def __repr__(self):
        return f'<DeliveryForm {self.id}>'    

class Delivery(db.Model):
    __tablename__ = 'deliveries'

    id = db.Column(db.Integer, primary_key=True)  
    order_id = db.Column(db.Integer, db.ForeignKey('order.id'), nullable=False) 
    delivery_address = db.Column(db.String(500), nullable=False)  
    delivery_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow) 
    delivery_status = db.Column(db.String(100), default="Pending")  
    delivery_person = db.Column(db.String(100), nullable=True)  

    # Relationship with the Order model
    order = db.relationship('Order', backref=db.backref('delivery', uselist=False))

    def __repr__(self):
        return f'<Delivery {self.id} - Order {self.order_id}>'

    def __init__(self, order_id, delivery_address, delivery_date, delivery_status="Pending", delivery_person=None):
        self.order_id = order_id
        self.delivery_address = delivery_address
        self.delivery_date = delivery_date
        self.delivery_status = delivery_status
        self.delivery_person = delivery_person

class DispatchRider(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    location = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20), nullable=False)
    vehicle_type = db.Column(db.String(50), nullable=False)  # Car or Bike
    vehicle_number = db.Column(db.String(50), nullable=False)
    vehicle_model = db.Column(db.String(100), nullable=False)
    vehicle_color = db.Column(db.String(50), nullable=False)
    vehicle_image = db.Column(db.String(200), nullable=True)  # Path to vehicle image
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    user = db.relationship('User', backref='dispatch_riders', lazy=True)

    def __repr__(self):
        return f'<DispatchRider {self.name}>'
    
#------------------------------------------JOBS--------------------------------------------------------------

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    employer_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    tags = db.Column(db.String(200), nullable=False)  # Comma-separated tags
    employer = db.relationship('User', backref=db.backref('Job', lazy=True))
    applications_count = db.Column(db.Integer, default=0)
    salary_range = db.Column(db.String(100), nullable=True) 


# JobApplication model
class JobApplication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=False)
    job_seeker_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=True)
    resume = db.Column(db.String(200), nullable=True)  # Path to resume file (optional)
    job = db.relationship('Job', backref=db.backref('applications', lazy=True))
    job_seeker = db.relationship('User', backref=db.backref('applications', lazy=True))

# Service model
class Service(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    provider_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    price = db.Column(db.Float, nullable=False)
    tags = db.Column(db.String(200), nullable=False)  # Comma-separated tags
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    provider = db.relationship('User', backref=db.backref('services', lazy=True))
    requests_count = db.Column(db.Integer, default=0)

# ServiceRequest model
class ServiceRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    service_id = db.Column(db.Integer, db.ForeignKey('service.id'), nullable=False)
    seeker_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=True)
    contact_details = db.Column(db.String(200), nullable=True)  # Contact info
    status = db.Column(db.String(50), default='Pending')  # Example status
    requested_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship definitions
    service = db.relationship('Service', backref=db.backref('requests', lazy=True))
    seeker = db.relationship('User', backref=db.backref('service_requests', lazy=True))


class Event(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=True)
    date = db.Column(db.DateTime, nullable=False)
    location = db.Column(db.String(255), nullable=False)
    bookings = db.relationship('EventBooking', backref='event', lazy=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False) 
    user = db.relationship('User', backref='events', lazy=True)
    def get_booking_count(self):
        return EventBooking.query.filter_by(event_id=self.id).count()

class EventBooking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    event_id = db.Column(db.Integer, db.ForeignKey('event.id'), nullable=False)
    booking_date = db.Column(db.DateTime, default=datetime.utcnow)
    special_notes = db.Column(db.String(500), nullable=True)
    user = db.relationship('User', backref='bookings')
    status = db.Column(db.String(20), default="Pending")  # e.g., Pending, Confirmed, etc.


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    content = db.Column(db.Text, nullable=True)
    media_url = db.Column(db.String(255), nullable=True)  # For images/videos
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    groups_id = db.Column(db.String, db.ForeignKey('groups.id'), nullable=True)
    user_id = db.Column(db.String, nullable=False)

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')   

# class MessageForm(FlaskForm):
#     content = StringField('Message', validators=[DataRequired()])
#     media = FileField('Media')  # For uploading images/videos
#     submit = SubmitField('Send')    

class Block(db.Model):
    __tablename__ = 'blocks'

    id = db.Column(db.Integer, primary_key=True)
    blocker_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    blocked_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    blocker_user = db.relationship('User', foreign_keys=[blocker_user_id])
    blocked_user = db.relationship('User', foreign_keys=[blocked_user_id])

    def __repr__(self):
        return f'<Block {self.blocker_user.username} blocked {self.blocked_user.username}>'
    
class Report(db.Model):
    __tablename__ = 'reports'

    id = db.Column(db.Integer, primary_key=True)
    reporter_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reported_user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    reason = db.Column(db.String(255), nullable=True)  # Reason for reporting
    
    reporter_user = db.relationship('User', foreign_keys=[reporter_user_id])
    reported_user = db.relationship('User', foreign_keys=[reported_user_id])

    def __repr__(self):
        return f'<Report by {self.reporter_user.username} against {self.reported_user.username}>' 

class Currency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(10), nullable=False, unique=True)  # e.g., "NGN", "USD"
    name = db.Column(db.String(50), nullable=False) 

    def __repr__(self):
        return f"<Currency {self.code}>"

class Ad(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    duration = db.Column(db.Integer, nullable=False)  # Duration in days
    payment_method = db.Column(db.String(50), nullable=False)
    media_url = db.Column(db.String(255))
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    currency_id = db.Column(db.Integer, db.ForeignKey('currency.id'), nullable=False)  # New field
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    payment_status = db.Column(db.String(50), default="Pending")

    currency = db.relationship('Currency', backref='ads')

    def __repr__(self):
        return f"<Ad {self.title}>"


class Storyteller(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    pen_name = db.Column(db.String(100), nullable=False)
    specialization = db.Column(db.String(50), nullable=False)
    bank_details = db.Column(db.String(255), nullable=False)
    paystack_public_key = db.Column(db.String(255), nullable=False)
    paystack_secret_key = db.Column(db.String(255), nullable=False)
    paystack_subaccount_code = db.Column(db.String(255), nullable=True)
    profile_picture = db.Column(db.String(255), nullable=True)  # Optional field


class Story(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    storyteller_id = db.Column(db.Integer, db.ForeignKey('storyteller.id'))
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    price = db.Column(db.Numeric(10, 2), nullable=False)
    is_series = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    earnings = db.Column(db.Float, default=0.0)
    story_type = db.Column(db.String(50), nullable=True)
    image = db.Column(db.LargeBinary)
    chunk_size = db.Column(db.Integer, default=200)

    @property
    def word_count(self):
        return len(self.content.split())
    
    def calculate_remaining_chunks(self, current_chunk_index):
        total_chunks = len(self.content.split()) // self.chunk_size
        return max(total_chunks - current_chunk_index, 0)


class Payment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    story_id = db.Column(db.Integer, db.ForeignKey('story.id'), nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)  # Which chunk was paid for
    amount = db.Column(db.Float, nullable=False)
    payment_date = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(50), nullable=False)


class PlatformEarnings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, default=0)

class PaymentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reference = db.Column(db.String(255), unique=True, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    storyteller_id = db.Column(db.Integer, db.ForeignKey('storyteller.id'), nullable=False)

class UserStoryProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    story_id = db.Column(db.Integer, db.ForeignKey('story.id'), nullable=False)
    words_read = db.Column(db.Integer, default=0)  # Tracks how many words the user has read
    chunk_index = db.Column(db.Integer, default=0)  # Tracks the current chunk index
    last_payment_time = db.Column(db.DateTime, default=datetime.utcnow)  # Tracks last payment time
   

#---------------------------------------------GROUPS---------------------------------------------


import uuid


class Group(db.Model):
    __tablename__ = 'groups'
    id = db.Column(db.String(36), primary_key=True, default=str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    rules = db.Column(db.Text, nullable=False)
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    profile_picture = db.Column(db.String(200), nullable=True)

class Membership(db.Model):
    __tablename__ = 'memberships'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    groups_id = db.Column(db.String(36), db.ForeignKey('groups.id'), nullable=True)
    status = db.Column(db.Enum('pending', 'approved', 'denied', name='membership_status'), default='pending')
    role = db.Column(db.String(20), default='member')  # Roles: 'admin', 'moderator', 'member'
    user = db.relationship('User', backref='memberships')
    groups = db.relationship('Group', backref='memberships')

class JoinRequest(db.Model):
    __tablename__ = 'join_requests'
    id = db.Column(db.Integer, primary_key=True)
    groups_id = db.Column(db.String, db.ForeignKey('groups.id'), nullable=True)
    user_id = db.Column(db.String, nullable=False)


#________________________________________________STARTUP TOOLKIT_______________________________________________________

class Investor(db.Model):
    __tablename__ = 'investors'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    preferences = db.Column(db.Text, nullable=True)
    email = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)  # Add the country column
    full_names = db.Column(db.String(255), nullable=True)  # New field for name
    occupation = db.Column(db.String(255), nullable=True) 
    

    user = db.relationship('User', backref='investors')
    
    def __init__(self, user_id, preferences, email,full_names,occupation, country):
        self.user_id = user_id
        self.preferences = preferences
        self.email = email
        self.country = country  # Make sure to initialize country here
        self.full_names=full_names
        self.occupation=occupation

class Cofounder(db.Model):
    __tablename__ = 'cofounders'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    preferences = db.Column(db.Text, nullable=True)
    email = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)  # Add the country column
    full_names = db.Column(db.String(255), nullable=True)  # New field for name
    occupation = db.Column(db.String(255), nullable=True) 
    
    user = db.relationship('User', backref='cofounders')
    
    def __init__(self, user_id, preferences, full_names, occupation,email, country):
        self.user_id = user_id
        self.preferences = preferences
        self.email = email
        self.country = country  # Initialize country here
        self.full_names=full_names
        self.occupation=occupation

class Startup(db.Model):
    __tablename__ = 'startups'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    idea = db.Column(db.Text, nullable=False)
    email = db.Column(db.String(100), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    country = db.Column(db.String(100), nullable=False)  # Add the country column
    full_names = db.Column(db.String(255), nullable=True)  # New field for name
    occupation = db.Column(db.String(255), nullable=True) 
    
    user = db.relationship('User', backref='startups')
    
    def __init__(self, user_id, idea, email,full_names, phone_number, country):
        self.user_id = user_id
        self.idea = idea
        self.email = email
        self.phone_number = phone_number
        self.country = country  # Initialize country here
        self.full_names=full_names




class Resource(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    file_path = db.Column(db.String(1024), nullable=False)



class MockModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        data = {
            'industry': ['Tech', 'Health', 'Food', 'Retail', 'Finance', 'Tech', 'Health'],
            'target_market': ['Young adults', 'Seniors', 'Millennials', 'Mass market', 'Small businesses', 'Tech enthusiasts', 'Health-conscious'],
            'success': [1, 1, 0, 0, 1, 1, 0]
        }
        df = pd.DataFrame(data)
        X = pd.get_dummies(df[['industry', 'target_market']])  # One-hot encoding
        y = df['success']
        self.model.fit(X, y)

    def predict(self, industry, target_market):
        X = pd.DataFrame([[industry, target_market]], columns=['industry', 'target_market'])
        X = pd.get_dummies(X)  # One-hot encoding
        prediction = self.model.predict(X)
        return prediction[0]
model = MockModel()

class Pitch(db.Model):
    __tablename__ = 'pitches'

    id = db.Column(db.Integer, primary_key=True)
    startup_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    investor_id = db.Column(db.Integer, db.ForeignKey('investors.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships (optional, for easier querying)
    startup = db.relationship('User', backref='pitches', lazy=True)
    investor = db.relationship('Investor', backref='pitches_received', lazy=True)

    def __repr__(self):
        return f"<Pitch {self.startup_id} -> {self.investors_id}>"
   

