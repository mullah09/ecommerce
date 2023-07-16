# Generated by Django 4.2 on 2023-05-03 16:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_alter_productreview_user'),
    ]

    operations = [
        migrations.CreateModel(
            name='Clickstream',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('item_id', models.PositiveIntegerField()),
                ('user_id', models.PositiveIntegerField(null=True)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]