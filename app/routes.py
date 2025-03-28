from werkzeug.security import check_password_hash, generate_password_hash
from flask import Blueprint, jsonify, send_from_directory, abort,render_template, request, redirect, Response, url_for, flash, session, current_app as app, current_app
from sqlalchemy.exc import IntegrityError
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
from pathlib import Path
from flask_socketio import emit, join_room,SocketIO, leave_room, send 
from .forms import MessageForm
from datetime import datetime, timedelta
from app import socketio
from playwright.sync_api import sync_playwright
import time
from pytrends.request import TrendReq
# from textblob import TextBlob
from .forms import BusinessIdeaForm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
# import numpy as np
# import redis

import pandas as pd
# from sklearn.metrics import classification_report
import pickle
import time
# from bs4 import BeautifulSoup
import nltk
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
from .models import db, model, User, Post,Pitch, Tag, Block, Report,UserStoryProgress,Investor, Startup, Cofounder, Resource,  Membership, Group, PaymentRecord, PlatformEarnings, Currency,Payment, Storyteller, Story, Message, Ad,EventBooking,Event,JobApplication,Comment,Follow,Job, Service, ServiceRequest, DispatchRider,Product,Delivery, Order, DeliveryForm # Correctly import db and User model


routes = Blueprint('routes', __name__)

# @routes.route('/')
# def home():
#     if 'username' in session:
#         return f"Welcome, {session['username']}!"
#     return render_template('intro.html') 

ON_RENDER = os.getenv("RENDER") is not None  # Detect if running on Render

def run_browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Ensure headless mode
        page = browser.new_page()
        page.goto("https://www.google.com")
        content = page.content()
        browser.close()
        return content

if not ON_RENDER:  # Prevent Playwright from running on Render
    print(run_browser())

@routes.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('routes.dashboard'))
    return render_template('intro.html')



MATRIX_SERVER = "https://matrix.org"  # Use public Matrix server

@routes.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            date_of_birth = request.form.get('date_of_birth', None)  # Optional
            country = request.form.get('country', None)  # Optional
            address = request.form.get('address', None)  # Optional
            id_verified = 'id_verified' in request.form
            accepted_terms = 'accepted_terms' in request.form

            if not username or not email or not password:  # Ensure required fields
                flash("Username, email, and password are required!", "danger")
                return redirect(url_for('routes.signup'))

            hashed_password = generate_password_hash(password)

            # Save user in your database
            new_user = User(
                username=username,
                email=email,
                password=hashed_password,
                date_of_birth=date_of_birth if date_of_birth else None,
                country=country if country else None,
                address=address if address else None,
                id_verified=id_verified,
                accepted_terms=accepted_terms,
                created_at=datetime.utcnow()
            )

            db.session.add(new_user)
            db.session.commit()

            # Register user on Matrix server
            try:
                matrix_response = requests.post(
                    f"{MATRIX_SERVER}/_matrix/client/r0/register",
                    json={
                        "username": username,  
                        "password": password,  
                        "auth": {"type": "m.login.dummy"}  # Bypass auth
                    }
                )
                if matrix_response.status_code == 200:
                    flash("Matrix account created successfully!", "success")
                else:
                    flash("Matrix registration failed, but your app account was created.", "warning")
            except Exception as e:
                flash("Could not connect to Matrix server.", "warning")

            flash("Account created successfully!", "success")
            return redirect(url_for('routes.login'))

        except IntegrityError:
            db.session.rollback()
            flash("Email already exists!", "danger")
            return redirect(url_for('routes.signup'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error: {str(e)}", "danger")  # Show full error
            return redirect(url_for('routes.signup'))

    return render_template('signup.html')


@routes.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username

            # Attempt Matrix login
            try:
                matrix_response = requests.post(
                    f"{MATRIX_SERVER}/_matrix/client/r0/login",
                    json={"type": "m.login.password", "user": user.username, "password": password}
                )
                if matrix_response.status_code == 200:
                    matrix_data = matrix_response.json()
                    session['matrix_token'] = matrix_data['access_token']
                    flash("Logged into Matrix successfully!", "success")
                else:
                    flash("Matrix login failed, but you can still use the app.", "warning")
            except Exception as e:
                flash("Could not connect to Matrix server.", "warning")

            flash("Logged in successfully!", "success")
            return redirect(url_for('routes.dashboard'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('routes.login'))

    return render_template('login.html')

@routes.route('/logout')
def logout():
    matrix_token = session.get('matrix_token')

    # Log out of Matrix if the token exists
    if matrix_token:
        try:
            requests.post(
                f"{MATRIX_SERVER}/_matrix/client/r0/logout",
                headers={"Authorization": f"Bearer {matrix_token}"}
            )
        except Exception as e:
            current_app.logger.error(f"Matrix logout failed: {e}")

    session.clear()
    flash("Logged out successfully!", "success")
    return redirect(url_for('routes.home'))

def get_logged_in_user():
    """
    Retrieve the logged-in user's details from the session.
    """
    user_id = session.get('user_id')  
    if not user_id:
        return None  
    user = User.query.get(user_id)
    return user

@routes.route('/profile', methods=['GET', 'POST'])
def edit_profile():
    if request.method == 'POST':
        # Retrieve user inputs
        current_user.name = request.form.get('username')
        current_user.email = request.form.get('email')
        current_user.bio = request.form.get('bio')

        # Handle profile image upload
        profile_image = request.files.get('profile_image')
        if profile_image and profile_image.filename:
            profile_filename = secure_filename(profile_image.filename)
            profile_image.save(os.path.join(app.config['UPLOAD_FOLDER'], profile_filename))
            current_user.profile_image = profile_filename

        # Handle banner image upload
        banner_image = request.files.get('banner_image')
        if banner_image and banner_image.filename:
            banner_filename = secure_filename(banner_image.filename)
            banner_image.save(os.path.join(app.config['UPLOAD_FOLDER'], banner_filename))
            current_user.banner_image = banner_filename

        # Save changes to the database
        try:
            db.session.commit()

            # Sync username with Matrix (if changed)
            matrix_token = session.get('matrix_token')
            if matrix_token:
                matrix_user_id = f"@{current_user.username}:matrix.org"
                requests.put(
                    f"{MATRIX_SERVER}/_matrix/client/r0/profile/{matrix_user_id}/displayname",
                    headers={"Authorization": f"Bearer {matrix_token}"},
                    json={"displayname": current_user.name}
                )

            flash('Profile updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {e}', 'danger')

        return redirect(url_for('routes.edit_profile'))

    return render_template('profile.html', user=current_user)

from collections import Counter

def calculate_trending_topics():
    tags = Tag.query.join(Post.tags).all()
    tag_count = Counter(tag.name for tag in tags)
    trending = tag_count.most_common(10)
    return [tag[0] for tag in trending] 

import re
from collections import Counter

# def calculate_trending_topics():
#     posts = Post.query.all()
#     post_scores = []
#     tag_scores = Counter()
    
#     for post in posts:
#         # Engagement-based scoring
#         score = post.likes * 2 + len(post.comments) + post.shares * 3
        
#         # Process tags
#         for tag in post.tags:
#             post_scores.append((tag.name, score))

#     # Filter and return top 10 trending topics
#     trending = tag_scores.most_common(10)
#     return [tag[0] for tag in trending]


def calculate_trending_topics():
    posts = Post.query.all()
    post_scores = []
    for post in posts:
        score = post.likes * 2 + len(post.comments) + post.shares * 3
        for tag in post.tags:
            post_scores.append((tag.name, score))
    tag_scores = Counter()
    for tag, score in post_scores:
        tag_scores[tag] += score

    trending = tag_scores.most_common(10)
    return [tag[0] for tag in trending]  

# def calculate_trending_from_content():
#     posts = Post.query.all()
#     word_list = []
#     stopwords = {'the', 'and', 'is', 'to', 'a', 'of', 'in', 'for', 'on', 'with', 'at', 'by'}
    
#     for post in posts:
#         words = re.findall(r'\w+', post.content.lower())
#         word_list.extend(words)
    
#     # Filter stopwords and count occurrences
#     word_count = Counter(word for word in word_list if word not in stopwords)
#     trending = word_count.most_common(10)
#     return [word[0] for word in trending]



@routes.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    jobs = Job.query.all()
    event = Event.query.all()
    user = get_logged_in_user()
    if not user:
        return redirect(url_for('routes.login'))

    # Fetch all posts and trending topics
    posts = Post.query.order_by(Post.created_at.desc()).all()
    trending_topics = calculate_trending_topics()
    follow_suggestions = suggest_users_to_follow(user)

    # Fetch active ads
    now = datetime.utcnow()
    active_ads = Ad.query.filter(
        Ad.start_date <= now,
        Ad.end_date >= now,
        Ad.payment_status == "Paid"
    ).all()

    return render_template(
        'dashboard.html',
        user=user,
        posts=posts,
        trending_topics=trending_topics,
        follow_suggestions=follow_suggestions,
        active_ads=active_ads,
        event=event,
        jobs=jobs
    )


from random import sample

from random import choices

def suggest_users_to_follow(user):
    # Get the list of followed user IDs
    followed_users_ids = [follow.followed_user_id for follow in user.followed_users]

    # Get users who are NOT followed
    if followed_users_ids:
        users_to_suggest = User.query.filter(User.id.notin_(followed_users_ids)).all()
    else:
        users_to_suggest = User.query.all()

    users_to_suggest = sorted(users_to_suggest, key=lambda u: len(u.posts) if u.posts else 0, reverse=True)

    suggested_users = choices(users_to_suggest, k=min(5, len(users_to_suggest)))

    return suggested_users




@routes.route('/create_post', methods=['POST'])
def create_post():
    user = get_logged_in_user()
    if not user:
        return redirect(url_for('routes.login'))

    content = request.form.get('content')
    tag_names = request.form.getlist('tags')  # List of tag names from the form

    if not content:
        flash("Post content cannot be empty!", "danger")
        return redirect(url_for('routes.dashboard'))

    new_post = Post(content=content, user_id=user.id)

    for tag_name in tag_names:
        tag = Tag.query.filter_by(name=tag_name).first()
        if not tag:
            tag = Tag(name=tag_name)
        new_post.tags.append(tag)

    db.session.add(new_post)
    db.session.commit()

    flash("Post created successfully!", "success")
    return redirect(url_for('routes.dashboard'))

@routes.route('/add_comment/<int:post_id>', methods=['POST'])
def add_comment(post_id):
    user = get_logged_in_user()
    if not user:
        return redirect(url_for('routes.login'))

    content = request.form.get('content')

    if not content:
        flash("Comment cannot be empty!", "danger")
        return redirect(url_for('routes.dashboard'))

    post = Post.query.get_or_404(post_id)
    comment = Comment(content=content, user_id=user.id, post_id=post.id)

    db.session.add(comment)
    db.session.commit()

    flash("Comment added successfully!", "success")
    return redirect(url_for('routes.dashboard'))

# @routes.route('/post/<int:post_id>', methods=['GET'])
# def view_post(post_id):
#     user = get_logged_in_user()
#     post = Post.query.get_or_404(post_id)
#     comments = Comment.query.filter_by(post_id=post.id).all()

#     # Fetch all active ads
#     now = datetime.utcnow()
#     active_ads = Ad.query.filter(Ad.start_date <= now, Ad.end_date >= now, Ad.payment_status == "Paid").all()

#     # Randomize ads
#     random_ads = random.sample(active_ads, min(len(active_ads), 1)) if active_ads else []

#     return render_template(
#         'dashboard.html',
#         post=post,
#         comments=comments,
#         active_ads=random_ads,
#         user=user
#     )

@routes.route('/following')
def following():
    user = get_logged_in_user() 
    if not user:
        return redirect(url_for('routes.login'))
    following_users = User.query.join(Follow, Follow.followed_user_id == User.id)\
                                .filter(Follow.follower_user_id == user.id).all()

    return render_template('following.html', following_users=following_users)

@routes.route('/follow/<int:user_id>', methods=['POST'])
def follow_user(user_id):
    user = get_logged_in_user() 
    if user.id == user_id:
        flash('You cannot follow yourself!', 'warning')
        return redirect(url_for('routes.user_profile', user_id=user_id))
    
    existing_follow = Follow.query.filter_by(
        follower_user_id=user.id,
        followed_user_id=user_id
    ).first()
    
    if not existing_follow:
        follow = Follow(follower_user_id=user.id, followed_user_id=user_id)
        db.session.add(follow)
        db.session.commit()
        update_blue_tick(user_id)  # Check blue tick status
        flash('You are now following this user!', 'success')
    else:
        flash('You are already following this user.', 'info')
    
    return redirect(url_for('routes.user_profile', user_id=user_id))

@routes.route('/unfollow/<int:user_id>', methods=['POST'])
def unfollow(user_id):
    user = get_logged_in_user() 
    follow = Follow.query.filter_by(
        follower_user_id=user.id,
        followed_user_id=user_id
    ).first()
    
    if follow:
        db.session.delete(follow)
        db.session.commit()
        update_blue_tick(user_id)  # Check blue tick status
        flash('You have unfollowed this user.', 'info')
    else:
        flash('You are not following this user.', 'warning')
    
    return redirect(url_for('routes.user_profile', user_id=user_id))

def update_blue_tick(user_id):
    follower_count = Follow.query.filter_by(followed_user_id=user_id).count()
    user = User.query.get(user_id)
    
    if follower_count >= 1000:
        if not user.blue_tick:  # Only update if it wasn't already assigned
            user.blue_tick = True
            db.session.commit()
    else:
        if user.blue_tick:  # Remove blue tick if followers drop below threshold
            user.blue_tick = False
            db.session.commit()


# @routes.route('/unfollow/<int:user_id>', methods=['POST'])
# def unfollow(user_id):
#     user = get_logged_in_user() 
#     if not user:
#         return redirect(url_for('routes.login'))
#     follow = Follow.query.filter_by(follower_user_id=user.id, followed_user_id=user_id).first()
#     if follow:
#         db.session.delete(follow)
#         db.session.commit()

    return redirect(url_for('routes.following'))


@routes.route('/user/<int:user_id>', methods=['GET', 'POST'])
def user_profile(user_id):
    user = User.query.get_or_404(user_id)
    logged_in_user = get_logged_in_user()  # Assuming a helper function to get the logged-in user
    
    if not logged_in_user:
        return redirect(url_for('routes.login'))
    
    # Check if the logged-in user is following the profile user
    is_following = Follow.query.filter_by(follower_user_id=logged_in_user.id, followed_user_id=user.id).first()

    # Handle follow/unfollow actions
    if request.method == 'POST':
        if 'follow' in request.form:
            # Follow the user
            if not is_following:
                follow = Follow(follower_user_id=logged_in_user.id, followed_user_id=user.id)
                db.session.add(follow)
                db.session.commit()
        elif 'unfollow' in request.form:
            # Unfollow the user
            if is_following:
                db.session.delete(is_following)
                db.session.commit()

    # Get list of users the current user is following (optional)
    following_users = User.query.join(Follow, Follow.followed_user_id == User.id)\
                                .filter(Follow.follower_user_id == logged_in_user.id).all()

    return render_template('user_profile.html', user=user, following_users=following_users, is_following=is_following)


@routes.route('/block/<int:user_id>', methods=['POST'])
def block_user(user_id):
    logged_in_user = get_logged_in_user()
    if not logged_in_user:
        return redirect(url_for('routes.login'))

    user_to_block = User.query.get_or_404(user_id)

    # Check if the user is already blocked
    existing_block = Block.query.filter_by(blocker_user_id=logged_in_user.id, blocked_user_id=user_to_block.id).first()

    if not existing_block:
        block = Block(blocker_user_id=logged_in_user.id, blocked_user_id=user_to_block.id)
        db.session.add(block)
        db.session.commit()

    return redirect(url_for('routes.user_profile', user_id=user_id))

@routes.route('/report/<int:user_id>', methods=['POST'])
def report_user(user_id):
    logged_in_user = get_logged_in_user()
    if not logged_in_user:
        return redirect(url_for('routes.login'))

    user_to_report = User.query.get_or_404(user_id)
    # You can customize the report with additional fields like a reason
    report = Report(reporter_user_id=logged_in_user.id, reported_user_id=user_to_report.id, reason="Inappropriate behavior")
    db.session.add(report)
    db.session.commit()

    return redirect(url_for('routes.user_profile', user_id=user_id))

@routes.route('/explore/trending')
def explore_trending():
    trending_topics = calculate_trending_topics()
    posts = Post.query.filter(Post.tags.any(Tag.name.in_(trending_topics))).order_by(Post.created_at.desc()).all()
    return render_template(
        'explore.html',
        posts=posts,
        trending_topics=trending_topics,
        active_menu='trending'
    )


@routes.route('/explore/sports')
def explore_sports():
    sports_tags = [
        'clubs', 'european league', 'cup', 'world cup',
        'messi', 'ronaldo', 'manchester', 'football', 'signed'
    ]
    posts = Post.query.filter(Post.tags.any(Tag.name.in_(sports_tags))).order_by(Post.created_at.desc()).all()
    return render_template(
        'explore.html',
        posts=posts,
        active_menu='sports'
    )



@routes.route('/explore/technology')
def explore_technology():
    technology_tags= [
        Post.tags.any(Tag.name == 'AI'),
        Post.tags.any(Tag.name == 'python'),
        Post.tags.any(Tag.name == 'software engineer'),
        Post.tags.any(Tag.name == 'cybersecurity'),
        Post.tags.any(Tag.name == 'programmer'),
        Post.tags.any(Tag.name == 'computer'),
        Post.tags.any(Tag.name == 'technology'),
        Post.tags.any(Tag.name == 'digital'),
        Post.tags.any(Tag.name == 'video editor')
    ]
    posts = Post.query.filter(Post.tags.any(Tag.name.in_(technology_tags))).order_by(Post.created_at.desc()).all()
    return render_template(
        'explore.html',
        posts=posts,
        active_menu='technology'
    )

from sqlalchemy.orm import aliased
from sqlalchemy import or_

@routes.route('/explore/politics')
def explore_politics():
    politics_tags=[
        Post.tags.any(Tag.name == 'politics'),
        Post.tags.any(Tag.name == 'state'),
        Post.tags.any(Tag.name == 'country'),
        Post.tags.any(Tag.name == 'president'),
        Post.tags.any(Tag.name == 'governor'),
        Post.tags.any(Tag.name == 'election'),
        Post.tags.any(Tag.name == 'policy'),
        Post.tags.any(Tag.name == 'Constitution'),
        Post.tags.any(Tag.name == 'leaders')
    ]
    posts = Post.query.filter(Post.tags.any(Tag.name.in_(politics_tags))).order_by(Post.created_at.desc()).all()
    return render_template(
        'explore.html',
        posts=posts,
        active_menu='politics'
    )

@routes.route('/explore')
def explore():
    return redirect(url_for('routes.explore_trending'))

#------------------------------------------------MARKETPLACE---------------------------------------------------------

@routes.route('/marketplace')
def marketplace():
    # Retrieve user ID from session
    user_id = session.get('user_id')
    if not user_id:
        flash('You must be logged in to access the marketplace.', 'danger')
        return redirect(url_for('routes.login'))  # Adjust 'routes.login' to your actual login route
    user = User.query.get(user_id)
    if not user:
        flash('User not found. Please log in again.', 'danger')
        return redirect(url_for('routes.login'))

    products = Product.query.all()  # Fetch all products
    orders = Order.query.filter_by(user_id=user.id).all()  # Fetch orders for the logged-in user
    return render_template('marketplace.html', products=products, orders=orders)

import random
from sqlalchemy import func

@routes.route('/product/<int:product_id>', methods=['GET'])
def product_detail(product_id):
    product = Product.query.get_or_404(product_id)
    similar_products = Product.query.filter(Product.name.ilike(f'%{product.name}%')).limit(5).all()
    recommended_products = Product.query.order_by(func.random()).limit(4).all()
    return render_template('product_detail.html', 
                           product=product, 
                           similar_products=similar_products, 
                           recommended_products=recommended_products)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mov'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@routes.route('/marketplace/create', methods=['GET', 'POST'])
def create_product():
    user_id = session.get('user_id')  
    if not user_id:
        flash("You must be logged in to create a product.", "danger")
        return redirect(url_for("routes.login"))

    # Ensure the upload directory exists
    upload_folder = Path(app.config['UPLOAD_FOLDER'])
    upload_folder.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        price = float(request.form['price'])
        tags = request.form['tags']
        quantity = int(request.form['quantity'])

        # Handle image uploads
        images = []
        for key in ['image_1', 'image_2', 'image_3']:
            image_file = request.files.get(key)
            if image_file and allowed_file(image_file.filename):
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                images.append(image_path)  # Add image path to list

        # Create new product with images list
        new_product = Product(
            name=name,
            description=description,
            price=price,
            quantity=quantity,
            tags=tags,
            images=images,  # Store image paths as a list
            user_id=user_id
        )

        try:
            db.session.add(new_product)
            db.session.commit()
            flash('Product created successfully!', 'success')
            return redirect(url_for('routes.marketplace'))
        except Exception as e:
            db.session.rollback()
            flash('Error creating product, please try again.', 'danger')

    return render_template('create_product.html')


@routes.route('/marketplace/edit/<int:product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    user_id = session.get('user_id')  # Get user_id from session
    if not user_id:
        flash("You must be logged in to edit a product.", "danger")
        return redirect(url_for("routes.login"))

    product = Product.query.get_or_404(product_id)
    if product.user_id != user_id:
        flash('You are not authorized to edit this product.', 'danger')
        return redirect(url_for('routes.marketplace'))

    if request.method == 'POST':
        product.name = request.form['name']
        product.description = request.form['description']
        product.price = float(request.form['price'])
        product.image_url = request.form['image_url']

        try:
            db.session.commit()
            flash('Product updated successfully!', 'success')
            return redirect(url_for('routes.product_detail', product_id=product.id))
        except Exception as e:
            db.session.rollback()
            flash('Error updating product, please try again.', 'danger')

    return render_template('edit_product.html', product=product)




@routes.route('/marketplace/order/<int:product_id>', methods=['POST'])
def place_order(product_id):
    user_id = session.get('user_id')  # Get user_id from session
    if not user_id:
        flash("You must be logged in to place an order.", "danger")
        return redirect(url_for("routes.login"))

    product = Product.query.get_or_404(product_id)
    quantity = int(request.form['quantity'])  # Ensure quantity is an integer

    order = Order(
        product_id=product.id,
        user_id=user_id,
        quantity=quantity,
        status='Pending'
    )

    try:
        db.session.add(order)
        db.session.commit()
        flash('Order placed successfully!', 'success')
        return redirect(url_for('routes.order_detail', order_id=order.id))
    except Exception as e:
        db.session.rollback()
        flash('Error placing order, please try again.', 'danger')



@routes.route('/marketplace/order/detail/<int:order_id>')
def order_detail(order_id):
    user_id = session.get('user_id')  # Get user_id from session
    if not user_id:
        flash("You must be logged in to view order details.", "danger")
        return redirect(url_for("routes.login"))

    order = Order.query.get_or_404(order_id)
    if order.user_id != user_id:
        flash('You are not authorized to view this order.', 'danger')
        return redirect(url_for('routes.marketplace'))

    return render_template('order_detail.html', order=order)



@routes.route('/marketplace/inventory/<int:product_id>', methods=['POST'])
def update_inventory(product_id):
    product = Product.query.get_or_404(product_id)
    if product.owner_id != current_user.id:
        flash('You are not authorized to update this product.', 'danger')
        return redirect(url_for('routes.marketplace'))

    product.stock = int(request.form['stock']) 

    try:
        db.session.commit()
        flash('Inventory updated successfully!', 'success')
        return redirect(url_for('routes.product_detail', product_id=product.id))
    except Exception as e:
        db.session.rollback()
        flash('Error updating inventory, please try again.', 'danger')



@routes.route('/marketplace/delivery/<int:order_id>', methods=['GET', 'POST'])
def handle_delivery(order_id):
    order = Order.query.get_or_404(order_id)
    if order.user_id != current_user.id:
        flash('You are not authorized to handle delivery for this order.', 'danger')
        return redirect(url_for('routes.marketplace'))

    # Get the dispatch riders available in the seller's location
    dispatch_riders = DispatchRider.query.filter_by(location=current_user.location).all()

    if request.method == 'POST':
        delivery_address = request.form['delivery_address']
        delivery_date = request.form['delivery_date']
        delivery_person_id = request.form.get('delivery_person')  # Dispatch rider selected by seller

        # Create a new delivery object and assign a dispatch rider
        delivery = Delivery(
            order_id=order.id, 
            delivery_address=delivery_address, 
            delivery_date=delivery_date,
            delivery_person_id=delivery_person_id  # Linking dispatch rider
        )
        db.session.add(delivery)
        db.session.commit()

        order.status = 'Shipped'  
        try:
            db.session.commit()
            flash('Delivery details saved successfully!', 'success')
            return redirect(url_for('routes.order_detail', order_id=order.id))
        except Exception as e:
            db.session.rollback()
            flash('Error saving delivery details, please try again.', 'danger')

    return render_template('delivery_form.html', order=order, dispatch_riders=dispatch_riders)




@routes.route('/delivery/<int:order_id>', methods=['GET', 'POST'])
def view_delivery(order_id):
    order = Order.query.get_or_404(order_id)

    if request.method == 'POST':
        delivery_address = request.form['delivery_address']
        delivery_person = request.form.get('delivery_person')  # Dispatch rider selected
        delivery_date = request.form['delivery_date']

        new_delivery = Delivery(
            order_id=order.id,
            delivery_address=delivery_address,
            delivery_date=delivery_date,
            delivery_person_id=delivery_person  # Link dispatch rider
        )

        try:
            db.session.add(new_delivery)
            db.session.commit()
            flash('Delivery details added successfully!', 'success')
            return redirect(url_for('routes.view_delivery', order_id=order.id))
        except Exception as e:
            db.session.rollback()
            flash('There was an error adding delivery details.', 'danger')

    delivery = order.delivery
    dispatch_rider = None
    if delivery and delivery.delivery_person_id:
        dispatch_rider = DispatchRider.query.get(delivery.delivery_person_id)

    return render_template('delivery_details.html', order=order, delivery=delivery, dispatch_rider=dispatch_rider)



@routes.route('/delivery/<int:delivery_id>/update_status', methods=['POST'])
def update_delivery_status(delivery_id):
    delivery = Delivery.query.get_or_404(delivery_id)
    if delivery.order.user_id != current_user.id and not current_user.is_admin:
        flash('You are not authorized to update this delivery status.', 'danger')
        return redirect(url_for('routes.dashboard'))

    new_status = request.form['delivery_status']
    delivery.delivery_status = new_status

    try:
        db.session.commit()
        flash('Delivery status updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash('There was an error updating the delivery status.', 'danger')

    return redirect(url_for('routes.view_delivery', order_id=delivery.order.id))

@routes.route('/orders', methods=['GET'])
def view_orders():
    user_id = session.get('user_id')  # Get user_id from session
    if not user_id:
        flash("You must be logged in to view orders.", "danger")
        return redirect(url_for("routes.login"))

    orders = Order.query.filter_by(user_id=user_id).all()
    pending_orders = [order for order in orders if order.status == 'Pending']
    finished_orders = [order for order in orders if order.status == 'Finished']

    return render_template(
        'orders.html',
        pending_orders=pending_orders,
        finished_orders=finished_orders
    )


@routes.route('/marketplace/seller_orders', methods=['GET'])
def seller_orders():
    user_id = session.get('user_id')  # Get user_id from session
    if not user_id:
        flash("You must be logged in to view orders.", "danger")
        return redirect(url_for("routes.login"))

    orders = Order.query.filter_by(user_id=user_id).all()
    pending_orders = [order for order in orders if order.status == 'Pending']
    finished_orders = [order for order in orders if order.status == 'Finished']

    return render_template(
        'sellers.html',
        pending_orders=pending_orders,
        finished_orders=finished_orders
    )

# sellers route for handling the seller's page
@routes.route('/sellers', methods=['GET'])
def sellers():
    user_id = session.get('user_id')  # Get the user_id from session

    if not user_id:
        flash("You must be logged in to access the seller's dashboard.", "danger")
        return redirect(url_for("routes.login"))
    user = User.query.get_or_404(user_id)
    
    products = Product.query.filter_by(user_id=user_id).all()
    orders = Order.query.filter(Order.product_id.in_([product.id for product in products])).all()
    
    # Organize orders by their status
    pending_orders = [order for order in orders if order.status == 'Pending']
    completed_orders = [order for order in orders if order.status == 'Completed']
    
    # Pass data to the template
    return render_template(
        'sellers.html',
        user=user,
        products=products,
        pending_orders=pending_orders,
        completed_orders=completed_orders
    )

@routes.route('/seller/products', methods=['GET', 'POST'])
def all_products():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to view your products.", "danger")
        return redirect(url_for('routes.login'))

    # Fetch all products created by the logged-in seller
    seller_products = Product.query.filter_by(user_id=user_id).all()

    if request.method == 'POST':
        product_id = request.form.get('product_id')  # Get product ID
        action = request.form.get('action')  # Action to perform

        if action == 'edit':
            # Fetch product to edit
            product = Product.query.get(product_id)
            if not product or product.user_id != user_id:
                flash("Unauthorized access to product.", "danger")
                return redirect(url_for('routes.all_products'))

            # Update product details
            product.name = request.form['name']
            product.description = request.form['description']
            product.price = float(request.form['price'])
            product.quantity = int(request.form['quantity'])
            product.tags = request.form['tags']

            # Handle banner type
            banner = request.form.get('banner')  # promo, sold, discount
            product.banner = banner if banner in ['promo', 'sold', 'discount' , 'new_arrival'] else None

            # Handle new images
            images = []
            for key in ['image_1', 'image_2', 'image_3']:
                image_file = request.files.get(key)
                if image_file and allowed_file(image_file.filename):
                    filename = secure_filename(image_file.filename)
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(image_path)
                    images.append(filename)
            if images:
                product.images = images  # Overwrite existing images

            try:
                db.session.commit()
                flash("Product updated successfully!", "success")
            except Exception as e:
                db.session.rollback()
                flash("Failed to update product. Please try again.", "danger")

        elif action == 'delete':
            # Delete a product
            product = Product.query.get(product_id)
            if not product or product.user_id != user_id:
                flash("Unauthorized access to delete product.", "danger")
                return redirect(url_for('routes.all_products'))

            try:
                db.session.delete(product)
                db.session.commit()
                flash("Product deleted successfully!", "success")
            except Exception as e:
                db.session.rollback()
                flash("Failed to delete product. Please try again.", "danger")

        return redirect(url_for('routes.all_products'))

    return render_template('sellers.html', products=seller_products)

@routes.route('/product/<int:product_id>', methods=['GET'])
def view_product(product_id):
    product = Product.query.get_or_404(product_id)
    product.visits += 1
    db.session.commit()
    return render_template('product.html', product=product)

@routes.route('/analytics', methods=['GET'])
def analytics():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to view analytics.", "danger")
        return redirect(url_for('routes.login'))
    total_orders = Order.query.filter_by(user_id=user_id).count()
    successful_deliveries = Order.query.filter_by(user_id=user_id, status='delivered').count()
    product_visits = db.session.query(func.sum(Product.visits)).filter_by(user_id=user_id).scalar() or 0
    monthly_data = (
        db.session.query(func.extract('month', Order.created_at).label('month'), func.count(Order.id).label('order_count'))
        .filter(Order.user_id == user_id)
        .group_by('month')
        .order_by('month')
        .all()
    )

    monthly_orders = {int(month): count for month, count in monthly_data}

    return render_template(
        'analytics.html',
        total_orders=total_orders,
        successful_deliveries=successful_deliveries,
        product_visits=product_visits,
        monthly_orders=monthly_orders
    )

#-----------------------------------------dispatch riders---------------------------------------

@routes.route('/dispatch_rider/register', methods=['GET', 'POST'])
def register_dispatch_rider():
    user_id = session.get('user_id')  
    if not user_id:
        flash("You must be logged in to register a dispatch rider.", "danger")
        return redirect(url_for('routes.login')) 

    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        location = request.form['location']
        email = request.form['email']
        phone = request.form['phone']
        vehicle_type = request.form['vehicle_type']
        vehicle_number = request.form['vehicle_number']
        vehicle_model = request.form['vehicle_model']
        vehicle_color = request.form['vehicle_color']
        vehicle_image = request.files.get('vehicle_image')
        filename = None
        if vehicle_image and allowed_file(vehicle_image.filename):
            filename = secure_filename(vehicle_image.filename)
            image_path = os.path.join('static/uploads', filename)
            vehicle_image.save(image_path)
        new_dispatch_rider = DispatchRider(
            name=name,
            address=address,
            location=location,
            email=email,
            phone=phone,
            vehicle_type=vehicle_type,
            vehicle_number=vehicle_number,
            vehicle_model=vehicle_model,
            vehicle_color=vehicle_color,
            vehicle_image=filename,
            user_id=user_id
        )

        try:
            db.session.add(new_dispatch_rider)
            db.session.commit()
            flash('Dispatch rider registered successfully!', 'success')
            return redirect(url_for('routes.view_all_dispatch_riders'))  
        except Exception as e:
            db.session.rollback()
            flash('There was an error registering the dispatch rider. Please try again.', 'danger')

    return render_template('register_dispatch_rider.html')



@routes.route('/seller/dispatch_riders', methods=['GET'])
def view_dispatch_riders():
    user_id = session.get('user_id')
    if not user_id:
        flash('You must be logged in to view dispatch riders.', 'danger')
        return redirect(url_for('routes.login'))  
    seller = User.query.get(user_id)
    if not seller:
        flash('User not found.', 'danger')
        return redirect(url_for('routes.marketplace')) 
    filtered_dispatch_riders = DispatchRider.query.filter_by(location=seller.address).all()
    all_dispatch_riders = DispatchRider.query.all()

    return render_template(
        'view_dispatch_rider.html',
        dispatch_riders=filtered_dispatch_riders,
        all_dispatch_riders=all_dispatch_riders
    )

#-----------------------------------------------------JOBS-----------------------------------------------------------

# Route to post a job
@routes.route('/post_job', methods=['GET', 'POST'])
def post_job(job_id):
    user_id = session.get('user_id')
    job = Job.query.get_or_404(job_id)
    if not user_id:
        flash("You must be logged in to post a job.", "danger")
        return redirect(url_for('routes.login'))

    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        tags = request.form['tags']
        salary_range = request.form.get('salary_range')  

        job = Job(
            title=title,
            description=description,
            employer_id=user_id,
            tags=tags,
            salary_range=salary_range
        )
        try:
            db.session.add(job)
            db.session.commit()
            flash("Job posted successfully!", "success")
            return redirect(url_for('routes.view_jobs'))
        except Exception as e:
            db.session.rollback()
            flash(f"Error posting job: {e}", "danger")
    return render_template('post_job.html', job=job)


@routes.route('/apply_job/<int:job_id>', methods=['GET', 'POST'])
def apply_job(job_id):
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to apply.", "danger")
        return redirect(url_for('routes.login'))

    job = Job.query.get_or_404(job_id)
    if request.method == 'POST':
        message = request.form['message']
        resume = request.files.get('resume')
        resume_path = None
        if resume:
            # Ensure the filename is secure
            from werkzeug.utils import secure_filename
            filename = secure_filename(resume.filename)
            resume_path = os.path.join('static', 'uploads', filename)
            # Save the resume
            resume.save(resume_path)

        application = JobApplication(
            job_id=job.id,
            job_seeker_id=user_id,
            message=message,
            resume=resume_path  # Store relative path in the database
        )
        job.applications_count += 1
        db.session.add(application)
        db.session.commit()
        flash("Application submitted successfully!", "success")
        return redirect(url_for('routes.view_jobs'))
    return render_template('apply_job.html', job=job)


RESUME_DIRECTORY = os.path.join(os.getcwd(), 'static', 'uploads')

@routes.route('/download_resume/<filename>')
def download_resume(filename):
    try:
        file_path = os.path.join(RESUME_DIRECTORY, filename)
        if not os.path.isfile(file_path):
            abort(404)  
        return send_from_directory(RESUME_DIRECTORY, filename, as_attachment=True)

    except Exception as e:
        flash(f"Error downloading file: {e}", "danger")
        return redirect(url_for('routes.view_applications')) 

# Route to post a service
@routes.route('/post_service', methods=['GET', 'POST'])
def post_service():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to post a service.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        tags = request.form['tags']
        price = request.form['price']
        service = Service(title=title, description=description, provider_id=user_id, price=price, tags=tags)
        try:
            db.session.add(service)
            db.session.commit()
            flash("Service posted successfully!", "success")
            return redirect(url_for('view_services'))
        except:
            db.session.rollback()
            flash("Error posting service.", "danger")
    return render_template('post_service.html')

# Route to view services
@routes.route('/view_services', methods=['GET', 'POST'])
def view_services():
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to view services.", "danger")
        return redirect(url_for('login'))

    search_query = request.args.get('search', '')
    services = Service.query.filter(Service.title.contains(search_query) | Service.tags.contains(search_query)).all()

    # Check if the user selected a service
    if request.method == 'POST':
        service_id = request.form.get('service_id')
        if service_id:
            # Store the selected service_id in the session
            session['service_id'] = service_id

    return render_template('view_services.html', services=services)


# Route for service seekers to request a service
@routes.route('/request_service/<int:service_id>', methods=['GET', 'POST'])
def request_service(service_id):
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to request a service.", "danger")
        return redirect(url_for('login'))

    service = Service.query.get_or_404(service_id)
    if request.method == 'POST':
        message = request.form['message']
        contact_details = request.form['contact_details']
        service_request = ServiceRequest(service_id=service.id, seeker_id=user_id, message=message, contact_details=contact_details)
        service.requests_count += 1
        db.session.add(service_request)
        db.session.commit()
        flash("Service request submitted successfully!", "success")
        return redirect(url_for('view_services'))
    return render_template('request_service.html', service=service)


@routes.route('/trending_jobs')
def trending_jobs():
    jobs = Job.query.order_by(Job.applications_count.desc()).limit(5).all()
    return render_template('trending_jobs.html', jobs=jobs)

@routes.route('/trending_services')
def trending_services():
    services = Service.query.order_by(Service.requests_count.desc()).limit(5).all()
    return render_template('trending_services.html', services=services)


@routes.route('/job', methods=['GET'])
def job():
    """Redirect users to the job dashboard"""
    return redirect(url_for('routes.job_dashboard'))


@routes.route('/job_dashboard', methods=['GET'])
def job_dashboard():
    # Ensure the user is logged in
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to access your job dashboard.", "danger")
        return redirect(url_for('routes.login'))  # Redirect to login if not logged in
    
    # Fetch job, user info, and related data
    jobs = Job.query.all()  # Get the job using job_id
    event = Event.query.all()
    user = User.query.get(user_id)

    # Fetch all jobs, services, and service requests
    jobs = Job.query.all()
    services = Service.query.all()
    service_requests = ServiceRequest.query.all()

    # Combine the posts in a unified format for easy rendering
    posts = [
        {'type': 'Job', 'title': job.title, 'description': job.description, 'id': job.id} for job in jobs
    ] + [
        {'type': 'Service', 'title': service.title, 'description': service.description, 'id': service.id} for service in services
    ] + [
        {'type': 'Service Request', 'title': request.title, 'description': request.description, 'id': request.id} for request in service_requests
    ]

    # Render the template with the necessary context
    return render_template('job_dashboard.html', posts=posts, user_id=user.id, event=event, jobs=jobs, username=user.username)

@routes.route('/view_jobs', methods=['GET'])
def view_jobs():
    events = Event.query.all()

    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to view jobs.", "danger")
        return redirect(url_for('routes.login'))
    jobs = []
    services = []
    service_requests = []
    search_query = request.args.get('search', '').strip()
    if search_query:
        jobs = Job.query.filter(Job.title.ilike(f"%{search_query}%")).all()
    else:
        jobs = Job.query.all()

    services = Service.query.all()
    service_requests = ServiceRequest.query.all()
    trending_jobs = Job.query.order_by(Job.applications_count.desc()).limit(5).all()
    trending_services = Service.query.order_by(Service.requests_count.desc()).limit(5).all()

    return render_template(
        'view_jobs.html',
        jobs=jobs,
        events=events,
        services=services,
        service_requests=service_requests,
        trending_jobs=trending_jobs,
        trending_services=trending_services
    )

@routes.route('/view_applications/<int:job_id>', methods=['GET'])
def view_applications(job_id):
    user_id = session.get('user_id')
    if not user_id:
        flash("You must be logged in to view applications.", "danger")
        return redirect(url_for('routes.login'))  

    job = Job.query.get_or_404(job_id)  
    if job.employer_id != user_id:
        flash("You are not authorized to view applications for this job.", "danger")
        return redirect(url_for('routes.job_dashboard'))  

    applications = JobApplication.query.filter_by(job_id=job.id).all()  # Query for applications

    return render_template('view_applications.html', job=job, applications=applications)


@routes.route('/view_service_requests', methods=['GET'])
def view_service_requests():
    user_id = session.get('user_id')
    service_id = session.get('service_id')  # Retrieve service_id from the session

    if not user_id or not service_id:
        flash("You must be logged in and have a service selected to view requests.", "danger")
        return redirect(url_for('routes.login'))  # Or redirect to a relevant page

    service = Service.query.get_or_404(service_id)

    if service.provider_id != user_id:
        flash("You are not authorized to view service requests for this service.", "danger")
        return redirect(url_for('routes.service_dashboard'))

    service_requests = ServiceRequest.query.filter_by(service_id=service.id).all()

    return render_template('view_service_requests.html', service=service, service_requests=service_requests)



#----------------------------------------------EVENTS----------------------------------------------------------


@routes.route('/book_event/<int:event_id>', methods=['GET', 'POST'])
def book_event(event_id):
    event = Event.query.get_or_404(event_id)
    print("Session user_id:", session.get('user_id'))

    if request.method == 'POST':
        booking_date = request.form['booking_date']
        special_notes = request.form['special_notes']

        if 'user_id' not in session:
            flash('You need to log in to book an event.', 'danger')
            return redirect(url_for('routes.login'))  # Redirect to login if not logged in

        new_booking = EventBooking(
            user_id=session['user_id'],  # Get user_id from session
            event_id=event.id,
            booking_date=datetime.strptime(booking_date, "%Y-%m-%dT%H:%M"),
            special_notes=special_notes
        )
        db.session.add(new_booking)
        db.session.commit()

        flash('Event booked successfully!', 'success')
        return redirect(url_for('routes.dashboard'))

    return render_template('book_event.html', event=event)



@routes.route('/event_detail/<int:event_id>')
def event_detail(event_id):
    event = Event.query.filter_by(id=event_id).first()  # Get the event by event_id
    bookings = EventBooking.query.filter_by(event_id=event_id).all()
    if event is None:
        flash('Event not found', 'danger')
        return redirect(url_for('routes.dashboard'))  # Redirect if event is not found
    print(event.id)
    return render_template('event_detail.html', event=event, event_id=event_id, bookings=bookings )




@routes.route('/create_event', methods=['GET', 'POST'])
def create_event():
    if 'user_id' not in session:
        flash('You must be logged in to create an event.', 'danger')
        return redirect(url_for('routes.login'))

    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        date = request.form['date']  # Expected format: "YYYY-MM-DDTHH:MM"
        location = request.form['location']

        # Parse the date correctly
        parsed_date = datetime.strptime(date, "%Y-%m-%dT%H:%M")

        new_event = Event(
            title=title,
            description=description,
            date=parsed_date,
            location=location,
            created_by=session['user_id']  # Storing the current logged-in user's ID
        )

        db.session.add(new_event)
        db.session.commit()
        flash('Event created successfully!', 'success')
        return redirect(url_for('routes.dashboard'))

    return render_template('create_event.html')


@routes.route('/events')
def all_events():
    events = Event.query.all()  # Fetch all events from the database
    return render_template('all_events.html', events=events)


#--------------------------------------------Messaging/Chats-------------------------------------------------
# Helper function for saving media
def save_media(file):
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return file_path
    return None


MATRIX_SERVER = "https://matrix.org"  # Public Matrix server

@routes.route('/messages', methods=['GET', 'POST'])
def messages():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    form = MessageForm()
    users = User.query.filter(User.id != user_id).all()
    recipient_id = request.args.get('user_id', type=int)
    current_chat_user = None
    messages = []

    # Use the stored Matrix token from session
    access_token = session.get('matrix_token')
    if not access_token:
        flash("You need to log in again.", "error")
        return redirect(url_for('login'))

    if recipient_id:
        current_chat_user = User.query.get(recipient_id)

        # Generate a unique DM room
        room_id = get_or_create_dm_room(user_id, recipient_id, access_token)

        if not room_id:
            flash("Could not join or create chat room.", "error")
            return redirect(url_for('messages'))

        # Ensure the user joins the room
        requests.post(
            f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/join",
            headers={"Authorization": f"Bearer {access_token}"}
        )

        # Get message history
        messages_response = requests.get(
            f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/messages",
            headers={"Authorization": f"Bearer {access_token}"}
        )
        if messages_response.status_code == 200:
            messages = messages_response.json().get("chunk", [])

    # Handle sending messages
    if form.validate_on_submit():
        if not recipient_id:
            flash("Recipient not selected.", "error")
            return redirect(url_for('messages'))

        content = form.content.data

        # Send message to Matrix
        send_response = requests.post(
            f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/send/m.room.message",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"msgtype": "m.text", "body": content}
        )

        if send_response.status_code == 200:
            flash("Message sent successfully!", "success")
        else:
            flash("Message could not be sent.", "error")

        return redirect(url_for('messages', user_id=recipient_id))

    return render_template(
        'messages.html',
        form=form,
        users=users,
        messages=messages,
        current_chat_user=current_chat_user

    )

def get_or_create_dm_room(user1, user2, access_token):
    """
    Get or create a direct message room for two users.
    """
    room_alias = f"dm_{min(user1, user2)}_{max(user1, user2)}"
    room_alias_full = f"#{room_alias}:matrix.org"

    # Check if the room already exists
    response = requests.get(
        f"{MATRIX_SERVER}/_matrix/client/r0/directory/room/{room_alias_full}",
        headers={"Authorization": f"Bearer {access_token}"}
    )

    if response.status_code == 200:
        return response.json().get("room_id")

    # Create a new room if it doesn’t exist
    create_response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/createRoom",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"room_alias_name": room_alias, "preset": "trusted_private_chat", "invite": [f"@{user2}:matrix.org"]}
    )

    if create_response.status_code == 200:
        return create_response.json().get("room_id")

    return None

def send_direct_message(sender, recipient, content, access_token):
    """
    Send a direct Matrix message (to-device messaging).
    """
    response = requests.put(
        f"{MATRIX_SERVER}/_matrix/client/r0/sendToDevice/m.room.message/{int(time.time())}",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "messages": {
                f"@{recipient}:matrix.org": {
                    "*": {  # Wildcard for all recipient devices
                        "msgtype": "m.text",
                        "body": content
                    }
                }
            }
        }
    )
    return response.json()

@routes.route('/matrix/read/<room_id>/<event_id>', methods=['POST'])
def mark_as_read(room_id, event_id):
    """
    Mark a message as read in Matrix.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403

    response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/receipt/m.read/{event_id}",
        headers={"Authorization": f"Bearer {session['matrix_token']}"}
    )

    return response.json(), response.status_code

@routes.route('/matrix/typing/<room_id>', methods=['POST'])
def typing_status(room_id):
    """
    Update typing status in a Matrix room.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403

    data = request.json
    typing = data.get("typing", False)
    matrix_user_id = session.get('matrix_user_id')

    if not matrix_user_id:
        return {"error": "Matrix user ID missing"}, 400

    response = requests.put(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/typing/{matrix_user_id}",
        headers={"Authorization": f"Bearer {session['matrix_token']}"},
        json={"timeout": 3000 if typing else 0}  # 3s timeout
    )

    return response.json(), response.status_code


#======================================================GROUP===============================================
MATRIX_SERVER = "https://matrix.org"  # Public Matrix server

@routes.route('/matrix/group_chat/<group_id>', methods=['GET'])
def group_chat(group_id):
    """
    Fetch messages for a group chat.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403

    access_token = session['matrix_token']
    room_id = get_or_create_group_room(group_id, access_token)

    if not room_id:
        return {"error": "Group chat room not found"}, 404

    response = requests.get(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/messages",
        headers={"Authorization": f"Bearer {access_token}"}
    )

    if response.status_code == 200:
        return response.json().get("chunk", [])
    return {"error": "Failed to fetch messages"}, response.status_code


@routes.route('/matrix/group_send/<group_id>', methods=['POST'])
def send_group_message(group_id):
    """
    Send a message to a Matrix group room.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403

    data = request.json
    content = data.get("content")
    access_token = session['matrix_token']
    room_id = get_or_create_group_room(group_id, access_token)

    if not room_id:
        return {"error": "Group chat room not found"}, 404

    response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/send/m.room.message",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"msgtype": "m.text", "body": content}
    )

    return response.json(), response.status_code


def get_or_create_group_room(group_id, access_token):
    """
    Get or create a Matrix room for a group chat.
    """
    room_alias = f"group_{group_id}"
    room_alias_full = f"#{room_alias}:matrix.org"

    response = requests.get(
        f"{MATRIX_SERVER}/_matrix/client/r0/directory/room/{room_alias_full}",
        headers={"Authorization": f"Bearer {access_token}"}
    )

    if response.status_code == 200:
        return response.json().get("room_id")

    create_response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/createRoom",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"room_alias_name": room_alias, "preset": "private_chat"}
    )

    if create_response.status_code == 200:
        return create_response.json().get("room_id")
    
    return None


  

@routes.route('/matrix/create_group', methods=['POST'])
def create_group():
    """
    Create a new Matrix group chat.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403

    data = request.json
    group_name = data.get("group_name")
    user_ids = data.get("user_ids", [])  # List of user IDs to invite
    
    access_token = session['matrix_token']
    
    create_response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/createRoom",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "name": group_name,
            "preset": "private_chat",
            "invite": [f"@{user}:matrix.org" for user in user_ids]
        }
    )
    
    return create_response.json(), create_response.status_code


@routes.route('/matrix/group_add_member/<room_id>', methods=['POST'])
def add_group_member(room_id):
    """
    Add a user to a Matrix group chat.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403
    
    data = request.json
    user_id = data.get("user_id")
    
    access_token = session['matrix_token']
    
    response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/invite",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"user_id": f"@{user_id}:matrix.org"}
    )
    
    return response.json(), response.status_code


@routes.route('/matrix/group_remove_member/<room_id>', methods=['POST'])
def remove_group_member(room_id):
    """
    Remove a user from a Matrix group chat (Admin only).
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403
    
    data = request.json
    user_id = data.get("user_id")
    
    access_token = session['matrix_token']
    
    response = requests.post(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/kick",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"user_id": f"@{user_id}:matrix.org"}
    )
    
    return response.json(), response.status_code


@routes.route('/matrix/group_members/<room_id>', methods=['GET'])
def list_group_members(room_id):
    """
    List all members of a Matrix group chat.
    """
    if 'matrix_token' not in session:
        return {"error": "Unauthorized"}, 403
    
    access_token = session['matrix_token']
    
    response = requests.get(
        f"{MATRIX_SERVER}/_matrix/client/r0/rooms/{room_id}/members",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    return response.json(), response.status_code
  


    
#----------------------------------------------ADS-------------------------------------------------------------


# @routes.route('/ads/create', methods=['POST'])
# def create_ad():
#     data = request.json
#     required_fields = ["user_id", "title", "description", "media_url", "duration", "payment_method"]

#     if not all(field in data for field in required_fields):
#         return jsonify({"error": "Missing required fields"}), 400

#     # Calculate payment amount
#     days = data["duration"]
#     payment_amount = days * DAILY_RATE

#     # Create a new ad
#     new_ad = Ad(
#         user_id=data["user_id"],
#         title=data["title"],
#         description=data["description"],
#         media_url=data["media_url"],
#         duration=days,
#         payment_method=data["payment_method"],
#         payment_amount=payment_amount
#     )
#     db.session.add(new_ad)
#     db.session.commit()

#     return jsonify({"message": "Ad created successfully", "ad_id": new_ad.id, "payment_amount": payment_amount}), 201

import logging

DAILY_RATE = 2500 

logging.basicConfig(level=logging.DEBUG)


# @routes.route('/ads/create', methods=['POST'])
# def create_ad():
#     data = request.json
#     required_fields = ["user_id", "title", "description", "media_url", "duration", "payment_method"]

#     media_file = request.files.get('media')
#     media_url = ""
#     if media_file and allowed_file(media_file.filename):
#         filename = secure_filename(media_file.filename)
#         filepath = os.path.join(UPLOAD_FOLDER, filename)
#         try:
#             media_file.save(filepath)
#             media_url = f"/{filepath}"
#             logging.info(f"Media file uploaded: {media_url}")  # Log successful media upload
#         except Exception as e:
#             logging.error(f"Error saving media file: {e}")  # Log error if file upload fails
#             return jsonify({"error": "Error uploading media file"}), 500


#     if not all(field in data for field in required_fields):
#         return jsonify({"error": "Missing required fields"}), 400

#     # Calculate payment amount
#     days = data["duration"]
#     payment_amount = days * DAILY_RATE

#     # Create a new ad
#     new_ad = Ad(
#         user_id=data["user_id"],
#         title=data["title"],
#         description=data["description"],
#         media_url=data["media_url"],
#         duration=days,
#         payment_method=data["payment_method"],
#         payment_amount=payment_amount
#     )
#     db.session.add(new_ad)
#     db.session.commit()

#     return jsonify({"message": "Ad created successfully", "ad_id": new_ad.id, "payment_amount": payment_amount}), 201
# def calculate_payment_amount(duration):
#     price_per_day = 2500  # Adjust this value based on your pricing
#     return price_per_day * duration


@routes.route('/ad/create', methods=['GET', 'POST'])
def create():
    if request.method == 'GET':
        currencies = Currency.query.all()
        return render_template('ad.html', currencies=currencies)

    elif request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        duration = request.form.get('duration')
        payment_method = request.form.get('paymentMethod')
        media_file = request.files.get('media')
        missing_fields = []
        if not title:
            missing_fields.append('title')
        if not description:
            missing_fields.append('description')
        if not duration:
            missing_fields.append('duration')
        if not payment_method:
            missing_fields.append('paymentMethod')
        if not media_file:
            missing_fields.append('media')

        if missing_fields:
            return jsonify({"error": "Missing required fields", "missing_fields": missing_fields}), 400
        currency_mapping = {
            "paystack": "NGN",  # Nigerian Naira
            "dollar_account": "USD"  # US Dollar
        }

        currency_code = currency_mapping.get(payment_method)
        if not currency_code:
            return jsonify({"error": "Invalid payment method or currency mapping not found"}), 400

        currency = Currency.query.filter_by(code=currency_code).first()
        if not currency:
            return jsonify({"error": f"Currency {currency_code} not found"}), 400

        # Retrieve user ID from the session or request context
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not logged in"}), 401

        # Save media file if provided
        media_path = None
        if media_file and allowed_file(media_file.filename):
            filename = secure_filename(media_file.filename)
            media_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            media_file.save(media_path)

        # Calculate payment amount based on duration
        DAILY_RATE = 2500  # ₦2,500 per 24 hours
        payment_amount = int(duration) * DAILY_RATE

        # Save the ad to the database
        ad = Ad(
            title=title,
            description=description,
            duration=int(duration),
            payment_method=payment_method,
            currency=currency,  # Save currency information
            media_url=media_path,
            user_id=user_id,
            payment_status="Pending"
        )
        db.session.add(ad)
        db.session.commit()

        return jsonify({
            "message": "Ad created successfully",
            "ad_id": ad.id,
            "payment_method": payment_method,
            "payment_amount": payment_amount,
            "currency": currency.code
        }), 201

@routes.route('/ad/pay', methods=['POST'])
def pay_ad():
    data = request.json
    ad_id = data.get("ad_id")
    payment_method = data.get("payment_method")
    reference = data.get("reference")

    if not ad_id or not payment_method:
        logging.error("Missing ad_id or payment_method")
        return jsonify({"error": "Missing ad_id or payment_method"}), 400

    # Fetch the ad from the database
    ad = Ad.query.get(ad_id)
    if not ad:
        logging.error("Ad not found")
        return jsonify({"error": "Ad not found"}), 404

    if payment_method == "paystack":
        headers = {
            "Authorization": f"Bearer sk_live_9df0d3121b2d9af968dee67aaf0e6012cebbea2f",  # Replace with your Paystack secret key
            "Content-Type": "application/json"
        }
        url = f"https://api.paystack.co/transaction/verify/{reference}"
        response = request.get(url, headers=headers)
        result = response.json()

        if response.status_code == 200 and result['status'] and result['data']['status'] == 'success':
            ad.payment_status = "Paid"
            ad.start_date = datetime.utcnow()
            ad.end_date = ad.start_date + timedelta(days=ad.duration)
            db.session.commit()

            add_to_trending(ad)  # Mark as trending if necessary
            return jsonify({"message": "Payment verified successfully!"}), 200
        else:
            logging.error("Payment verification failed")
            return jsonify({"error": "Payment verification failed"}), 400

    elif payment_method == "paypal":
        logging.info("Simulating PayPal payment verification")
        ad.payment_status = "Pending Verification"
        db.session.commit()
        return jsonify({"message": "Pending PayPal payment verification."}), 200

    logging.error("Invalid payment method")
    return jsonify({"error": "Invalid payment method"}), 400

def add_to_trending(ad):
    tags = re.findall(r"#\w+", ad.description)
    for tag in tags:
        trending_ad = Tag(name=tag, ad_id=ad.id)
        db.session.add(trending_ad)
    db.session.commit()


@routes.route('/ad/cleanup', methods=['POST'])
def cleanup_ads():
    current_time = datetime.utcnow()
    expired_ads = Ad.query.filter(Ad.end_date < current_time, Ad.payment_status == "Paid").all()

    for ad in expired_ads:
        ad.payment_status = "Expired"
        db.session.commit()
    return jsonify({"message": f"{len(expired_ads)} ads deactivated."}), 200

@routes.route('/ad')
def ad():
    return render_template('ad.html')


#------------------------------------------------------STORY-------------------------------------------------------------------------------


@routes.route('/story/<int:story_id>', methods=['GET'])
def view_story(story_id):
    story = Story.query.get_or_404(story_id)
    user_id = session.get('user_id')
    payment = Payment.query.filter_by(user_id=user_id, story_id=story_id).order_by(Payment.chunk_index.desc()).first()
    unlocked_chunks = payment.chunk_index if payment else 0
    words = story.content.split()
    total_chunks = -(-len(words) // 200)  # Ceiling division for chunks
    remaining_chunks = max(total_chunks - unlocked_chunks, 0)
    start = unlocked_chunks * 200
    end = (unlocked_chunks + 1) * 200
    if start >= len(words):
        return jsonify({'error': 'You have unlocked the full story'}), 400
    content_chunk = " ".join(words[start:end])
    next_chunk_available = end < len(words)
    return jsonify({
        'title': story.title,
        'next_chunk_content': content_chunk,
        'next_chunk_available': next_chunk_available,
        'price': story.price if next_chunk_available else 0,  # Price for the next chunk
        'remaining_chunks': remaining_chunks,  # Remaining chunks to unlock
    })


@routes.route('/story', methods=['GET'])
def list_stories():
    stories = Story.query.all()
    user_id = session.get('user_id')

    remaining_chunks = {}
    for story in stories:
        payment = Payment.query.filter_by(user_id=user_id, story_id=story.id).order_by(Payment.chunk_index.desc()).first()
        unlocked_chunks = payment.chunk_index if payment else 0
        total_chunks = -(-len(story.content.split()) // 200) 
        remaining_chunks[story.id] = max(total_chunks - unlocked_chunks, 0)
    return render_template('story.html', stories=stories, remaining_chunks=remaining_chunks)


@routes.route('/storyteller/register', methods=['GET', 'POST'])
def storyteller_register():
    if request.method == 'POST':
        pen_name = request.form['pen_name']
        specialization = request.form['specialization']
        bank_details = request.form['bank_details']
        paystack_public_key = request.form['paystack_public_key']
        paystack_secret_key = request.form['paystack_secret_key']
        user_id = session.get('user_id')
        if not user_id:
            flash("Please log in to register as a storyteller.", "danger")
            return redirect('/login')
        profile_picture = request.files['profile_picture']
        if profile_picture and allowed_file(profile_picture.filename):
            filename = secure_filename(profile_picture.filename)
            profile_picture_path = os.path.join(UPLOAD_FOLDER, filename)
            profile_picture.save(profile_picture_path)
        else:
            flash("Invalid profile picture format. Please upload a valid image file.", "danger")
            return redirect('/storyteller/register')
        storyteller = Storyteller(
            user_id=user_id,
            pen_name=pen_name,
            specialization=specialization,
            bank_details=bank_details,
            paystack_public_key=paystack_public_key,
            paystack_secret_key=paystack_secret_key,
            profile_picture=profile_picture_path
        )
        try:
            db.session.add(storyteller)
            db.session.commit()
            flash("Storyteller account created successfully! Please log in to continue.", "success")
            return redirect(url_for('storyteller_login'))
        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {e}", "danger")
            return redirect('/storyteller/register')
    return render_template('storyteller_register.html')

@routes.route('/story')
def story():
    stories = Story.query.all()
    return render_template('story.html', stories=stories)

@routes.route('/storyteller/dashboard', methods=['GET'])
def storyteller_dashboard():
    user_id = session.get('users_id')
    if not user_id:
        return redirect('/login')

    storyteller = Storyteller.query.filter_by(user_id=user_id).first()
    if not storyteller:
        return redirect('/storyteller/register')

    stories = Story.query.filter_by(storyteller_id=storyteller.id).all()
    return render_template('storyteller_dashboard.html', storyteller=storyteller, stories=stories)


@routes.route('/storyteller/create', methods=['GET', 'POST'])
def create_story_or_series():
    user_id = session.get('users_id')
    storyteller = Storyteller.query.filter_by(user_id=user_id).first()

    if not storyteller:
        return jsonify({'error': 'Unauthorized access'}), 403

    if request.method == 'GET':
        story_type = request.args.get('type')  
        return render_template('storyteller_create.html', type=story_type)  

    # Handling POST request
    title = request.form.get('title')
    content = request.form.get('content')
    story_type = request.form.get('type')  
    image = request.files.get('image')

    if len(content.split()) < 200:
        return jsonify({'error': 'Content must be at least 200 words'}), 400
    image_data = image.read() if image else None
    story = Story(
        title=title,
        content=content,
        story_type=story_type,  # Ensuring `type` is captured correctly
        storyteller_id=storyteller.id,
        price=500,
        image=image_data,
    )

    try:
        db.session.add(story)
        db.session.commit()
        return jsonify({'success': 'Story created successfully'}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@routes.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        image_url = f'uploads/{filename}' 
        new_story = Story(title=request.form['title'], content=request.form['content'], image_url=image_url)
        db.session.add(new_story)
        db.session.commit()

        return redirect(url_for('story'))
    

@routes.route('/storyteller/edit/<int:story_id>', methods=['GET', 'POST'])
def storyteller_edit(story_id):
    user_id = session.get('users_id')
    storyteller = Storyteller.query.filter_by(user_id=user_id).first()
    if not storyteller:
        return redirect('/storyteller/register')

    story = Story.query.filter_by(id=story_id, storyteller_id=storyteller.id).first()
    if not story:
        return jsonify({'error': 'Story not found or unauthorized access'}), 404

    if request.method == 'POST':
        story.title = request.form['title']
        story.content = request.form['content']
        story.image_url = request.form.get('image_url', '')
        story.word_count = len(story.content.split())
        story.price = (story.word_count // 200) * 500
        db.session.commit()
        return jsonify({'success': 'Story/Series updated successfully!'}), 200

    return render_template('storyteller_edit.html', story=story)


@routes.route('/storyteller/delete/<int:story_id>', methods=['POST'])
def storyteller_delete(story_id):
    user_id = session.get('users_id')
    storyteller = Storyteller.query.filter_by(user_id=user_id).first()
    if not storyteller:
        return jsonify({'error': 'Unauthorized access'}), 403

    story = Story.query.filter_by(id=story_id, storyteller_id=storyteller.id).first()
    if not story:
        return jsonify({'error': 'Story not found or unauthorized access'}), 404

    db.session.delete(story)
    db.session.commit()
    return jsonify({'success': 'Story/Series deleted successfully!'}), 200

@routes.route('/edit/story/<int:story_id>', methods=['GET', 'POST'])
def edit_story(story_id):
    story = Story.query.get_or_404(story_id)

    if request.method == 'POST':
        try:
            new_title = request.form.get('title')
            new_content = request.form.get('content')

            if not new_title or not new_content:
                return jsonify({'success': False, 'error': 'Title and content cannot be empty'}), 400

            story.title = new_title
            story.content = new_content
            db.session.commit()

            return jsonify({'success': True, 'message': 'Story updated successfully'}), 200
        except Exception as e:
            db.session.rollback()  # Rollback any changes in case of an error
            print(f"Error updating story: {e}")
            return jsonify({'success': False, 'error': 'An error occurred while updating the story'}), 500
    return render_template('edit_story.html', story=story)


@routes.route('/delete/story/<int:story_id>', methods=['DELETE'])
def delete_story(story_id):
    story = Story.query.get_or_404(story_id)
    db.session.delete(story)
    db.session.commit()
    return jsonify({'success': 'Story deleted successfully'}), 200


@routes.route('/storyteller/earnings', methods=['GET'])
def storyteller_earnings():
    user_id = session.get('users_id')
    storyteller = Storyteller.query.filter_by(user_id=user_id).first()
    if not storyteller:
        return jsonify({'error': 'Unauthorized access'}), 403

    total_earnings = db.session.query(db.func.sum(Story.earnings)).filter_by(storyteller_id=storyteller.id).scalar() or 0
    return jsonify({'earnings': total_earnings})


@routes.route('/storyteller/earnings', methods=['GET'])
def get_earnings():
    user_id = session.get('users_id')
    storyteller = Storyteller.query.filter_by(user_id=user_id).first()

    if not storyteller:
        return jsonify({'error': 'Unauthorized access'}), 403

    total_earnings = storyteller.earnings
    return jsonify({'earnings': total_earnings})

from app.payments import PaystackAPI  

# @routes.route('/storyteller/payments', methods=['POST'])
# def process_payment():
#     data = request.json
#     story_id = data.get('story_id')
#     user_id = session.get('user_id')

#     if not user_id or not story_id:
#         return jsonify({'error': 'Invalid request'}), 400

#     story = Story.query.filter_by(id=story_id).first()
#     if not story:
#         return jsonify({'error': 'Story not found'}), 404
#     amount = story.price  
#     paystack_api = PaystackAPI(secret_key="your_paystack_secret_key_here")  
#     payment_response = paystack_api.initiate_payment(amount, user_id)

#     if payment_response['status'] == 'success': 
#         payout_to_storyteller = amount * 0.8  
#         payout_to_you = amount * 0.2  

#         storyteller = Storyteller.query.filter_by(user_id=user_id).first()
#         storyteller.earnings += payout_to_storyteller
#         db.session.commit()

#         # Record or update your earnings (this could be in a separate table for platform earnings)
#         # Assuming there's a platform model to track platform earnings (optional)
#         # platform_earnings = PlatformEarnings.query.first()
#         # platform_earnings.amount += payout_to_you
#         # db.session.commit()
#         return jsonify({'success': 'Payment processed successfully', 'storyteller_earnings': payout_to_storyteller, 'platform_earnings': payout_to_you}), 200

#     return jsonify({'error': 'Payment failed. Please try again.'}), 500



PLATFORM_SUBACCOUNT_CODE = "ACCT_1pqt3s5k3gax57u" 

@routes.route('/storyteller/payments', methods=['POST'])
def process_payment():
    data = request.json
    story_id = data.get('story_id')
    user_id = session.get('users_id')

    if not user_id or not story_id:
        return jsonify({'error': 'Invalid request'}), 400

    story = Story.query.filter_by(id=story_id).first()
    if not story:
        return jsonify({'error': 'Story not found'}), 404

    storyteller = Storyteller.query.filter_by(id=story.storyteller_id).first()
    if not storyteller:
        return jsonify({'error': 'Storyteller not found'}), 404

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    paystack_api = PaystackAPI(secret_key="your_platform_secret_key")
    payment_response = paystack_api.initiate_payment(
        amount=story.price,
        email=user.email,
        storyteller_subaccount=storyteller.paystack_subaccount_code,
        platform_subaccount=PLATFORM_SUBACCOUNT_CODE
    )

    if payment_response.get('status') == 'success':
        return jsonify({'success': 'Payment initiated successfully', 'payment_url': payment_response['data']['authorization_url']}), 200

    return jsonify({'error': 'Payment initiation failed'}), 500



@routes.route('/webhook', methods=['POST'])
def paystack_webhook():
    data = request.json

    if data.get('event') == 'charge.success':
        reference = data['data']['reference']
        verification_response = PaystackAPI(secret_key="your_platform_secret_key").verify_payment(reference)

        if verification_response['data']['status'] == 'success':
            amount_paid = verification_response['data']['amount'] / 100  # Convert from kobo to naira
            storyteller_subaccount = verification_response['data']['subaccount']
            storyteller_share = 0.8  
            platform_share = 0.2  
            if not PaymentRecord.query.filter_by(reference=reference).first():
                try:
                    storyteller = Storyteller.query.filter_by(paystack_subaccount_code=storyteller_subaccount).first()
                    if not storyteller:
                        return jsonify({'error': 'Storyteller not found for this subaccount'}), 404
                    storyteller.earnings += amount_paid * storyteller_share
                    platform_earnings = PlatformEarnings.query.first()
                    if not platform_earnings:
                        platform_earnings = PlatformEarnings(amount=0)
                        db.session.add(platform_earnings)
                    platform_earnings.amount += amount_paid * platform_share
                    payment_record = PaymentRecord(
                        reference=reference,
                        amount=amount_paid,
                        storyteller_id=storyteller.id
                    )
                    db.session.add(payment_record)
                    db.session.commit()

                    return jsonify({'success': 'Payment verified and processed'}), 200
                except Exception as e:
                    db.session.rollback()
                    return jsonify({'error': f'Failed to process payment: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Payment verification failed'}), 400

    return jsonify({'error': 'Invalid webhook event'}), 400


PAYSTACK_SECRET_KEY = "sk_live_9df0d3121b2d9af968dee67aaf0e6012cebbea2f" 
def create_storyteller_subaccount(bank_details, storyteller_name, storyteller_email, storyteller_share=80):
    url = "https://api.paystack.co/subaccount"
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "business_name": storyteller_name,
        "settlement_bank": bank_details['bank_code'],  # e.g., "044" for Access Bank
        "account_number": bank_details['account_number'], 
        "percentage_charge": 100 - storyteller_share,  
        "primary_contact_email": storyteller_email,
        "primary_contact_name": storyteller_name
    }
    response = request.post(url, headers=headers, json=data)
    if response.status_code == 201: 
        return response.json()['data']['subaccount_code']
    else:
        raise Exception(f"Failed to create subaccount: {response.json()}")
    
@routes.route('/register_storyteller', methods=['POST'])
def register_storyteller():
    data = request.json
    storyteller_name = data.get('pen_name')
    storyteller_email = data.get('email')
    bank_details = {
        'bank_code': data.get('bank_code'),
        'account_number': data.get('account_number'),
    }
    
    try:
        subaccount_code = create_storyteller_subaccount(bank_details, storyteller_name, storyteller_email)
        new_storyteller = Storyteller(
            user_id=session.get('users_id'),
            pen_name=storyteller_name,
            specialization=data.get('specialization'),
            bank_details=f"{bank_details['bank_code']}:{bank_details['account_number']}",
            paystack_subaccount_code=subaccount_code
        )
        db.session.add(new_storyteller)
        db.session.commit()
        
        return jsonify({'success': 'Storyteller registered successfully'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@routes.route('/storyteller/login', methods=['GET', 'POST'])
def storyteller_login():
    if request.method == 'POST':
        pen_name = request.form['pen_name']
        user_id = session.get('users_id') 
        storyteller = Storyteller.query.filter_by(pen_name=pen_name, user_id=user_id).first()
        if storyteller:
            session['storyteller_id'] = storyteller.id 
            return redirect('/storyteller/dashboard')  
        else:
            flash("Invalid pen name or unauthorized access. Please try again.", "error")
            return redirect('/storyteller/login') 

    return render_template('storyteller_login.html')



import uuid 
@routes.route('/story/pay/<int:story_id>', methods=['POST'])
def pay_for_story(story_id):
    try:
        data = request.get_json()
        if not data or 'chunk_index' not in data:
            return jsonify({'success': False, 'error': 'Missing chunk index'}), 400

        chunk_index = data['chunk_index']
        story = Story.query.filter_by(id=story_id).first()
        if not story:
            return jsonify({'success': False, 'error': 'Story not found'}), 404
        
        if not story.chunk_size or not isinstance(story.chunk_size, list):
            return jsonify({'success': False, 'error': 'Story chunks not defined or invalid'}), 500

        if chunk_index >= len(story.chunk_size):
            return jsonify({'success': False, 'error': 'No more story chunks available'}), 400

        try:
            amount = story.chunk_size[chunk_index]['price']
        except (KeyError, IndexError):
            return jsonify({'success': False, 'error': 'Invalid chunk pricing information'}), 500

        paystack_api = PaystackAPI(secret_key="sk_live_9df0d3121b2d9af968dee67aaf0e6012cebbea2f")
        payment_reference = f"{story_id}-{chunk_index}-{uuid.uuid4().hex}"

        payment_response = paystack_api.initialize_transaction(
            reference=payment_reference,
            amount=amount,
            email="customer_email@example.com"
        )

        if not payment_response.get('status'):
            return jsonify({'success': False, 'error': 'Failed to initiate payment'}), 500

        return jsonify({'success': True, 'payment_url': payment_response['data']['authorization_url']})

    except Exception as e:
        import traceback
        print(f"Error in pay_for_story: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': 'Internal server error'}), 500

@routes.route('/story/pay/<int:story_id>', methods=['POST'])
def pay_for_next_chunk(story_id):
    try:
        user_id = current_user.id
        progress = UserStoryProgress.query.filter_by(user_id=user_id, story_id=story_id).first()
        if not progress:
            print(f"Progress not found for user {user_id}, story {story_id}")
            return jsonify({'error': 'No progress found for this story.'}), 400
        story = Story.query.get(story_id)
        if not story or not story.content:
            print(f"Story {story_id} not found or has no content.")
            return jsonify({'error': 'Story not found or empty.'}), 404
        words_read = progress.words_read
        chunk_index = progress.chunk_index
        content_words = story.content.split()
        if words_read >= len(content_words):
            print(f"User {user_id} has already read the entire story {story_id}.")
            return jsonify({'error': 'You have unlocked the full story.'}), 400
        next_chunk_content = " ".join(content_words[words_read: words_read + 200])
        next_chunk_available = words_read + 200 < len(content_words)

        progress.words_read += 200  
        progress.chunk_index += 1
        db.session.commit()

        print(f"User {user_id} paid for chunk {chunk_index + 1} of story {story_id}.")
        return jsonify({
            'success': True,
            'next_chunk_content': next_chunk_content,
            'next_chunk_available': next_chunk_available
        })

    except Exception as e:
        import traceback
        print(f"Error in pay_for_next_chunk: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error occurred.'}), 500

def update_user_progress(user_id, story_id, words_read, chunk_index):
    progress = UserStoryProgress.query.filter_by(user_id=user_id, story_id=story_id).first()
    if progress:
        progress.words_read = words_read
        progress.chunk_index = chunk_index
        db.session.commit()
    else:
        new_progress = UserStoryProgress(user_id=user_id, story_id=story_id, words_read=words_read, chunk_index=chunk_index)
        db.session.add(new_progress)
        db.session.commit()

#----------------------------------------------------GROUPS-------------------------------------------------------------------        


@routes.route('/groups', methods=['GET'])
def groups():
    groups = Group.query.all()
    return render_template('groups.html', groups=groups)

@routes.route('/groups/<group_id>', methods=['GET'])
def view_group(group_id):
    groups = Group.query.get(group_id)
    if not groups:
        return "Group not found", 404
    posts = Post.query.filter_by(group_id=group_id).all()
    return render_template('group.html', groups=groups, posts=posts)

@routes.route('/groups/create', methods=['POST'])
def create_group():
    if 'users_id' not in session:
        return jsonify({"error": "User not authenticated"}), 401
    user_id = session['user_id']
    data = request.json
    if 'name' not in data or 'rules' not in data:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        groups = Group(name=data['name'], rules=data['rules'], owner_id=user_id)
        db.session.add(groups)
        db.session.commit()

        # Add the user as admin in the Membership table
        membership = Membership(user_id=user_id, groups_id=groups.id, status='approved', role='admin')
        db.session.add(membership)
        db.session.commit()

        # Return a success response
        return jsonify({"message": "Group created successfully", "redirect_url": url_for('routes.view_group', group_id=groups.id)})

    except Exception as e:
        # Log the exception for debugging
        app.logger.error(f"Error creating group: {str(e)}")
        return jsonify({"error": "Failed to create group"}), 500
    
@routes.route('/groups/<groups_id>/join', methods=['POST'])
def join_group(groups_id):
    user_id = session.get('users_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401
    existing_request = Membership.query.filter_by(user_id=user_id, groups_id=groups_id).first()
    if existing_request:
        return jsonify({"error": "Already requested or part of the group"}), 400
    request = Membership(user_id=user_id, groups_id=groups_id, status='pending')
    db.session.add(request)
    db.session.commit()

    return jsonify({"message": "Join request sent successfully"})

@routes.route('/groups/<groups_id>/manage_request', methods=['POST'])
def manage_request(groups_id):
    data = request.json
    target_user_id = data['users_id']
    action = data['action']

    # Validate admin
    membership = Membership.query.filter_by(user_id=current_user.id, groups_id=groups_id, role='admin').first()
    if not membership:
        return jsonify({"error": "Unauthorized"}), 403

    # Process request
    request_membership = Membership.query.filter_by(user_id=target_user_id, groups_id=groups_id, status='pending').first()
    if not request_membership:
        return jsonify({"error": "Request not found"}), 404

    if action == 'approve':
        request_membership.status = 'approved'
    elif action == 'deny':
        request_membership.status = 'denied'
    else:
        return jsonify({"error": "Invalid action"}), 400

    db.session.commit()
    return jsonify({"message": f"Request {action}ed successfully"})

@routes.route('/messages', methods=['GET'])
def view_messages():
    user_id = current_user.id
    messages = Message.query.filter((Message.sender_id == user_id) | (Message.receiver_id == user_id)).order_by(Message.timestamp.desc()).all()
    return render_template('messages.html', messages=messages)

@routes.route('/messages/send', methods=['POST'])
def send_message():
    data = request.json
    receiver_id = data.get('receiver_id')
    content = data.get('content')

    if not receiver_id or not content:
        return jsonify({"error": "Receiver and content are required"}), 400

    message = Message(sender_id=current_user.id, receiver_id=receiver_id, content=content)
    db.session.add(message)
    db.session.commit()

    return jsonify({"message": "Message sent successfully"})


@routes.route('/groups/<group_id>/manage_users', methods=['GET'])
def manage_users(group_id):
    group = Group.query.get(group_id)
    if not group:
        return "Group not found", 404

    if not Membership.query.filter_by(user_id=current_user.id, group_id=group_id, role='admin').first():
        return jsonify({"error": "Unauthorized"}), 403

    memberships = Membership.query.filter_by(group_id=group_id).all()
    return render_template('manage_users.html', group=group, memberships=memberships)

@routes.route('/groups/<group_id>/update_role', methods=['POST'])
def update_role(group_id):
    data = request.json
    user_id = data.get('users_id')
    new_role = data.get('role')

    if not Membership.query.filter_by(user_id=current_user.id, group_id=group_id, role='admin').first():
        return jsonify({"error": "Unauthorized"}), 403

    membership = Membership.query.filter_by(user_id=user_id, group_id=group_id).first()
    if not membership:
        return jsonify({"error": "Membership not found"}), 404

    if new_role not in ['admin', 'moderator', 'member']:
        return jsonify({"error": "Invalid role"}), 400

    membership.role = new_role
    db.session.commit()

    return jsonify({"message": f"User role updated to {new_role}"})


#__________________________________________________STARTUP TOOLKIT_____________________________________________

import requests

def payment_required(f):
    def scout(*args, **kwargs):  # New function name
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "User not logged in"}), 401
        user = User.query.get(user_id)
        if not user or not user.premium_status or (user.premium_expiry and user.premium_expiry < datetime.utcnow()):
            return jsonify({"error": "Premium access required"}), 403
        return f(*args, **kwargs)
    return scout


@routes.route('/subscribe', methods=['GET', 'POST'])
def subscribe_page():
    if request.method == 'GET':
        return render_template('subscribe.html')

    user_id = session.get('users_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    payment_data = {
        "email": user.email,
        "amount": 10000 * 100, 
        "callback_url": "http://localhost:5000/verify-payment"
    }
    headers = {
        "Authorization": f"Bearer YOUR_PAYSTACK_SECRET_KEY",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.paystack.co/transaction/initialize", json=payment_data, headers=headers)

    if response.status_code == 200:
        data = response.json()
        return jsonify({"payment_url": data["data"]["authorization_url"]})
    else:
        return jsonify({"error": "Payment initialization failed"}), 500


@routes.route('/verify-payment', methods=['POST'])
def verify_payment():
    payload = request.get_json()
    if not payload or payload.get("event") != "charge.success":
        return jsonify({"error": "Invalid payment verification request"}), 400

    payment_data = payload["data"]
    user = User.query.filter_by(email=payment_data["customer"]["email"]).first()

    if not user:
        return jsonify({"error": "User not found"}), 404
    user.premium_status = True
    user.premium_expiry = datetime.utcnow() + timedelta(days=30)
    payment = Payment(
        user_id=user.id,
        amount=payment_data["amount"] / 100,  # Convert kobo to naira
        status="success"
    )
    db.session.add(payment)
    db.session.commit()

    return jsonify({"message": "Payment verified successfully"}), 200



@routes.route('/register-role', methods=['GET', 'POST'])
def register_role():
    if request.method == 'GET':
        # Render the role registration page (role.html)
        return render_template('role.html')

    # Handle the POST request for role registration
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({"error": "User not logged in"}), 401

    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.json
    role = data.get('role')
    if role not in ['investor', 'cofounder', 'startup']:
        return jsonify({"error": "Invalid role"}), 400
    
    full_names = data.get('full_names')
    occupation = data.get('occupation')

    user.role = role
    db.session.commit()

    # Add specific details based on role
    if role == 'investor':
        preferences = data.get('preferences')
        email = data.get('email')
        country = data.get('country')  # Get country for investor
        investor = Investor(user_id=user.id, preferences=preferences, email=email, country=country, full_names=full_names, occupation=occupation)
        db.session.add(investor)

    elif role == 'cofounder':
        preferences = data.get('preferences')
        email = data.get('email')
        country = data.get('country')  # Get country for cofounder
        cofounder = Cofounder(user_id=user.id, preferences=preferences, email=email, country=country, full_names=full_names, occupation=occupation)
        db.session.add(cofounder)

    elif role == 'startup':
        idea = data.get('idea')
        email = data.get('email')
        phone_number = data.get('phone_number')
        country = data.get('country')  # Get country for startup
        startup = Startup(user_id=user.id, idea=idea, email=email, phone_number=phone_number, country=country, full_names=full_names, occupation=occupation)
        db.session.add(startup)

    db.session.commit()
    return jsonify({"message": f"Role '{role}' registered successfully"}), 200


@routes.route('/view-role', methods=['GET'])
def view_role():
    """
    View data for all roles. Everyone sees all users: investors, cofounders, and startups.
    """
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Fetch all relevant data
    investors = Investor.query.all()
    cofounders = Cofounder.query.all()
    startups = Startup.query.all()

    return render_template(
        'startup_view.html',
        investors=investors,
        cofounders=cofounders,
        startups=startups,
        premium=user.premium_status
    )


@routes.route('/investors', methods=['GET'], endpoint='investors')
@payment_required
def investors():
    investors = Investor.query.all()
    return render_template('investors.html', investors=investors)


@routes.route('/pitch', methods=['POST'], endpoint='pitch')
@payment_required
def pitch():
    """
    Allow startups to pitch to investors.
    """
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if not user or user.role != 'startup':
        return jsonify({"error": "Only startups can submit pitches"}), 403

    data = request.json
    investor_id = data.get('investor_id')
    investor = Investor.query.get(investor_id)

    if not investor:
        return jsonify({"error": "Investor not found"}), 404

    # Log the pitch (you can also notify the investor here)
    pitch = Pitch(
        startup_id=user.id,
        investor_id=investor_id,
        message=data.get('message'),
    )
    db.session.add(pitch)
    db.session.commit()

    return jsonify({"message": f"Pitch sent to {investor.name}"}), 200








#----------------------------------------------RESOURCES PAGE------------------------------------------------------#
@routes.route('/resources', methods=['GET'])
def resources():
    """
    Query all resources grouped by category and render the resources page.
    """
    resources_by_category = {}
    resources = Resource.query.all()

    for resource in resources:
        if resource.category not in resources_by_category:
            resources_by_category[resource.category] = []
        resources_by_category[resource.category].append({
            "title": resource.title,
            "description": resource.description,
            "id": resource.id,
            "file_url": url_for('static', filename=f"uploads/{os.path.basename(resource.file_path)}"),  # Generate URL
        })

    return render_template('resources.html', resources_by_category=resources_by_category)


ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

def allowed_file(filename):
    """
    Check if the uploaded file has a valid extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@routes.route('/upload_resource', methods=['GET', 'POST'], endpoint='upload_resource')
def upload_resource():
    """
    Handle resource upload by users.
    """
    if request.method == 'GET':
        return render_template('upload_resource.html')

    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        category = request.form.get('category')
        file = request.files.get('file')  # File input from form

        # Validate form fields
        if not title or not description or not category or not file:
            flash("All fields, including the file, are required.", "error")
            return render_template('upload_resource.html')

        # Validate file type
        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed types are: PDF, DOC, DOCX.", "error")
            return render_template('upload_resource.html')

        # Secure the file name and save it
        filename = secure_filename(file.filename)
        upload_folder = os.path.join(app.root_path, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)

        # Save the resource to the database
        new_resource = Resource(
            title=title,
            description=description,
            category=category,
            user_id=session.get('user_id'),
            file_path=f'static/uploads/{filename}'  # Store relative file path
        )
        db.session.add(new_resource)
        db.session.commit()

        flash("Resource uploaded successfully!", "success")
        return redirect(url_for('routes.resources'))



@routes.route('/delete_resource/<int:resource_id>', methods=['POST'])
def delete_resource(resource_id):
    """
    Delete a resource and its associated file.
    """
    resource = Resource.query.get(resource_id)

    if not resource:
        flash("Resource not found.", "error")
        return redirect(url_for('routes.resources'))

    # Get the file path
    file_path = os.path.join(app.root_path, resource.file_path)

    # Remove the file from the filesystem if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete the resource from the database
    db.session.delete(resource)
    db.session.commit()

    flash("Resource deleted successfully!", "success")
    return redirect(url_for('routes.resources'))

#--------------------------------------------------------------RESOURCE PAGE ENDS--------------------------------------#

#______________________________________________________INVESTORS______________________________________________________-#

# @routes.route('/pitch', methods=['POST'],endpoint='pitch' )
# @payment_required
# def pitch():
#     user_id = session.get('user_id')
#     user = User.query.get(user_id)

#     if not user or user.role != 'startup':
#         return jsonify({"error": "Only startups can submit pitches"}), 403

#     data = request.json
#     investor_id = data.get('investor_id')
#     investor = Investor.query.get(investor_id)

#     if not investor:
#         return jsonify({"error": "Investor not found"}), 404

#     return jsonify({"message": f"Pitch sent to {investor_id}"}), 200


# @routes.route('/investors', methods=['GET'],endpoint='investors' )
# @payment_required
# def investors():
#     investors = Investor.query.all()
#     return render_template('investors.html', investors=investors)


# Homepage Route for Submenu
@routes.route('/startup-toolkit')
def startup_toolkit():
    user_id = session.get('user_id')  
    
    if not user_id:
        return redirect(url_for('routes.dashboard')) 

    # Fetch the user by `id` (users.id)
    user = User.query.get(user_id)
    
    if not user:
        return "User not found", 404

    return render_template('startup_toolkit.html', user=user)



# @routes.route('/post-resource', methods=['POST'])
# @payment_required
# def post_resource():
#     user_id = session.get('user_id')
#     user = User.query.get(user_id)
    
#     if not user:
#         return jsonify({"error": "User not found"}), 404

#     data = request.json
#     category = data.get('category')
#     title = data.get('title')
#     description = data.get('description')
#     link = data.get('link')

#     # Ensure valid category
#     valid_categories = ['legal', 'business development', 'education', 'templates']
#     if category not in valid_categories:
#         return jsonify({"error": "Invalid category"}), 400

#     # Create the resource and store in database
#     resource = Resource(user_id=user.id, category=category, title=title, description=description, link=link)
#     db.session.add(resource)
#     db.session.commit()

#     return jsonify({"message": "Resource posted successfully"}), 200


#-------------------------------------------------BUSINESS IDEA----------------------------------------------------#



# Initialize Redis cache
ON_RENDER = os.getenv("RENDER") is not None  # Render sets certain environment variables

# Use Redis if available and not on Render
if not ON_RENDER:
    try:
        import redis
        cache = redis.StrictRedis(host='localhost', port=6379, db=0)
        cache.ping()  # Check if Redis is available
    except (ImportError, ConnectionError):
        ON_RENDER = True  # Fallback to dictionary-based cache
else:
    ON_RENDER = True

# Dictionary-based cache for Render or if Redis is unavailable
if ON_RENDER:
    class LocalCache:
        """Simple dictionary-based cache fallback."""
        def __init__(self):
            self.store = {}

        def set(self, key, value):
            self.store[key] = value

        def get(self, key):
            return self.store.get(key)

        def delete(self, key):
            self.store.pop(key, None)

    cache = LocalCache()

nltk.download("vader_lexicon")

def train_model():
    """
    Trains a logistic regression model to predict the success probability of business ideas 
    using realistic world data and market factors.
    """
    # Define training data
    X_train = pd.DataFrame([
        {"industry": "Tech", "targetMarket": "USA"},
        {"industry": "Healthcare", "targetMarket": "USA"},
        {"industry": "E-commerce", "targetMarket": "Europe"},
        {"industry": "Finance", "targetMarket": "Europe"},
        {"industry": "Agriculture", "targetMarket": "Africa"},
        {"industry": "Healthcare", "targetMarket": "Africa"},
        {"industry": "E-commerce", "targetMarket": "Asia"},
        {"industry": "Education", "targetMarket": "Asia"},
        {"industry": "Renewable Energy", "targetMarket": "Australia"},
        {"industry": "Mining", "targetMarket": "Australia"},
        {"industry": "Tech", "targetMarket": "South America"},
        {"industry": "Tourism", "targetMarket": "South America"},
        {"industry": "Manufacturing", "targetMarket": "North America"},
        {"industry": "Retail", "targetMarket": "North America"},
        {"industry": "Logistics", "targetMarket": "Middle East"},
        {"industry": "Real Estate", "targetMarket": "Middle East"}
    ])
    y_train = [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0]  # 1: Success, 0: Failure

    # Preprocessing pipeline
    column_transformer = ColumnTransformer(
        transformers=[
            ("industry", OneHotEncoder(handle_unknown="ignore"), ["industry"]),
            ("targetMarket", OneHotEncoder(handle_unknown="ignore"), ["targetMarket"]),
        ],
        remainder="drop"
    )

    model_pipeline = Pipeline(steps=[
        ("preprocessor", column_transformer),
        ("classifier", LogisticRegression())
    ])

    # Train the model
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

def cache_model(cache, model, key="business_idea_model"):
    """
    Cache the trained model.
    """
    serialized_model = pickle.dumps(model)
    cache.set(key, serialized_model)

def load_model_from_cache(cache, key="business_idea_model"):
    """
    Load the model from cache.
    """
    serialized_model = cache.get(key)
    if serialized_model:
        return pickle.loads(serialized_model)
    return None

def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text using NLTK's VADER.
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)["compound"]
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Train and cache the model
model = train_model()
cache_model(cache, model)

# Load model from cache for prediction
model_from_cache = load_model_from_cache(cache)
if model_from_cache:
    print("Model loaded from cache successfully!")
else:
    print("Failed to load model from cache.")

# Competitor search function using Playwright
def search_and_analyze_idea(idea, max_retries=3):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto('https://www.google.com')

            try:
                page.wait_for_selector('button[jsname="V67aGc"]', state="visible", timeout=10000)
                page.locator('button[jsname="V67aGc"]').click()  # Close cookie banner
            except Exception:
                pass            
            page.wait_for_selector('input[name="q"]', state="visible", timeout=10000)

            # Fill in the search input with the business idea
            page.fill('input[name="q"]', idea)
            page.keyboard.press("Enter")

            # Wait for the search results to load
            page.wait_for_selector('.g')

            # Extract competitor information from the search results
            search_results = page.locator('.g')
            competitor_data = []

            for result in search_results.all():
                competitor_name = result.locator('h3').text_content() or "Unknown"
                description = result.locator('.IsZvec').text_content() or "No description available"
                competitor_data.append({"name": competitor_name, "description": description})

            browser.close()

            if competitor_data:
                return {"competitors": competitor_data}
            else:
                return {"message": "No competitors found for this search term."}
    
    except Exception as e:
        return {"error": f"The request failed: {str(e)}"}
    
idea = "AI-powered business automation"
result = search_and_analyze_idea(idea)
print(result)    


# Google trends analysis using Playwright
def get_google_trends(idea, max_retries=3):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Example: Search trends for the idea
            page.goto('https://trends.google.com/trends/trendingsearches/daily')

            # Wait for the page to load and search the idea
            page.wait_for_selector('input[aria-label="Search"]', state="visible", timeout=10000)
            page.fill('input[aria-label="Search"]', idea)
            page.keyboard.press("Enter")

            # Wait for results to load
            page.wait_for_selector('.fe-trend')

            # Extract trend titles or other relevant information
            trends = page.locator('.fe-trend .title').all_text_contents()

            browser.close()

            if trends:
                return {"trends": trends}
            else:
                return {"message": "No trends data found for this search term."}

    except Exception as e:
        return {"error": f"The request failed: {str(e)}"}

# Business Idea Analysis
def analyze_business_idea(idea, industry, target_market):
    try:
        # Search competitors and trends based on the business idea
        competitor_info = search_and_analyze_idea(idea)
        competitor_summary = competitor_info.get("competitors", "No competitor data available")

        # Fetch trends for the business idea (optional)
        trends_data = get_google_trends(idea)
        trends_summary = trends_data if isinstance(trends_data, dict) else "No significant trends found."

        # Sentiment Analysis
        sentiment_result = analyze_sentiment(f"{idea} {industry}")

        # Success Prediction
        success_probability = model.predict(pd.DataFrame([{"industry": industry, "targetMarket": target_market}]))
        success = success_probability[0] == 1

        # Return the analysis result
        return {
            "success": success,
            "trendsAnalysis": trends_summary,
            "competitorAnalysis": competitor_summary,
            "sentiment": sentiment_result,
        }

    except Exception as e:
        return {"error": str(e)}

# Flask Route to Predict Business Idea
@routes.route('/predict_idea', methods=['GET', 'POST'])
def predict_idea():
    form = BusinessIdeaForm()
    if request.method == 'POST':
        if not form.validate_on_submit():
            return jsonify({"error": "Form validation failed. Please check your inputs."}), 400

        data = {
            "idea": form.idea.data,
            "industry": form.industry.data,
            "targetMarket": form.targetMarket.data,
        }

        try:
            result = analyze_business_idea(
                data["idea"], data["industry"], data["targetMarket"]
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        if "error" in result:
            return jsonify({"error": result["error"]}), 400

        response = {
            "success": bool(result["success"]),
            "marketOpportunity": result.get("marketOpportunity", "N/A"),
            "message": "Your business idea may face challenges." if not result["success"] else "Your business idea shows potential!",
            "strengths": result.get("strengths", []),
            "pitfalls": result.get("pitfalls", []),
            "trendsAnalysis": result.get("trendsAnalysis", "N/A"),
            "competitorAnalysis": result.get("competitorAnalysis", "N/A"),
            "sentiment": result.get("sentiment", "N/A"),
        }

        return jsonify(response)

    # Render form template for GET requests
    return render_template('business_idea_form.html', form=form, result_text=None)