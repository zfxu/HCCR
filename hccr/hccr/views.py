from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import json
import base64
from hccr.hccr_recognize import recognize_character

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def process(request):
    if (request.method == "POST") and (request.POST.get('id') == "1"):
        img_str = request.POST.get('txt')
        img_str = base64.b64decode(img_str)
        character, val = recognize_character(img_str)
        return HttpResponse(json.dumps({"status": 1, "char": character, "val": val}))
