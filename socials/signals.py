from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import SocialPost, PostMedia
from .helper import train_post_embedding, train_post_image_hash

@receiver(post_save, sender=SocialPost)
def handle_post_training(sender, instance, created, **kwargs):
    """
    Automatically triggers AI training (embedding generation) 
    whenever a social post is created or updated without a vector.
    """
    if created or not instance.vector:
        train_post_embedding(instance)


@receiver(post_save, sender=PostMedia)
def handle_media_image_hash(sender, instance, created, **kwargs):
    """
    Automatically generates pHash visual fingerprint for an image
    whenever a new PostMedia is saved.
    """
    if created and not instance.image_hash:
        train_post_image_hash(instance.post)
