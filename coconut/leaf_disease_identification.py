from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from django.http import JsonResponse


@csrf_exempt
def leaf_disease_identification(request):
    
    # if request.method == '':
    if  request.FILES['image']:
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_file_url = fs.url(filename)
        IMAGE_SIZE = 64

        model = load_model("C:/Users/anish/Documents/coconut/backend/coconut/coconut/coconut_model.h5")
        random_image = cv2.imread(fs.path(filename), cv2.IMREAD_GRAYSCALE)
        random_image = cv2.resize(random_image, (IMAGE_SIZE, IMAGE_SIZE))
        random_image = np.array(random_image)
        random_image = random_image.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
        random_image = random_image.astype('float32')
        random_image /= 255
        prediction = model.predict(random_image)
        if np.argmax(prediction) == 0:
            return JsonResponse({"hasDisease":"true"})
        else:
            return JsonResponse({"hasDisease":"false"})
        
        # return HttpResponse("Hello, worldsdf!")
    # Return a 400 Bad Request error if the request method is not POST or the 'image' field is missing
    # return HttpResponse("Hello,dsd world!")
    