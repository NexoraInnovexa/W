from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, FileField, SelectField
from wtforms.validators import DataRequired 
from wtforms import StringField, FileField, SubmitField
from app.models import User

# Define a form class for sending a message
class MessageForm(FlaskForm):
    recipient_id = SelectField('Recipient', coerce=int, validators=[DataRequired()])
    content = TextAreaField('Message', validators=[DataRequired()])
    media = FileField('Media (optional)')
    # email = StringField('Email', validators=[Email()])
    emoji = StringField('Emoji')
    submit = SubmitField('Send')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipient_id.choices = [(user.id, user.username) for user in User.query.all()]



class BusinessIdeaForm(FlaskForm):
    idea = TextAreaField("Describe Your Business Idea", validators=[DataRequired()])
    industry = StringField("Industry", validators=[DataRequired()])
    targetMarket = StringField("Target Market", validators=[DataRequired()])
    competition = StringField("Competition", validators=[DataRequired()])
