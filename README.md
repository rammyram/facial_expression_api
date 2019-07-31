# Facial_Expression_API
An API to detect facial expression's either by cam or by uploading image.

# Steps :
1. Install the requirements : pip install -r requirements.txt
2. Now all you have to do it is run : python api.py 
3. The app should be up and running at : http://localhost:5000
4. Valid end points are : http://localhost:5000/cam  => cam will open up and recognize your expression providing 1st two guesses 
5. You can modify predict_emotion function in api.py to format the prediction in required format.
6. We can also send an image to the API and get the result back : http://localhost:5000/img_upload
