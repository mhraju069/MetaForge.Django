from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import SocialPost
from .helper import train_post_embedding

@receiver(post_save, sender=SocialPost)
def handle_post_training(sender, instance, created, **kwargs):
    """
    Automatically triggers AI training (embedding generation) 
    whenever a social post is created or updated without a vector.
    """
    if created or not instance.vector:
        # Run the training logic
        train_post_embedding(instance)
