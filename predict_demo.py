from run_resnet_xgb import predict_single_image

image_path = r"C:\Users\AVICHAL TRIVEDI\Documents\Citrus Disease Classification using ResNet50 + XGBoost\Orange Dataset\random google images for presentation\cropped fresh.png"
   # the image to check and find percentage
result = predict_single_image(image_path, epochs_to_use=10)
print(result)
