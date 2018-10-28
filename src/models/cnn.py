img_path = '../../dataset/asl_alphabet_test/A_test.jpg'


img = image.load_img(img_path)
print(type(img))
x = image.img_to_array(img)

plt.imshow(x[:, :, 2])
plt.show()