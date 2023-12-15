# realllllllllllllffffinalalllllll

## Data Preprocessing
데이터의 차원 축소를 위해 PCA(Principal Component Analysis) 수행  
pca500은 KNN을 위해, pca1000은 SVM을 위해 사용 

    pca = PCA(n_components=500, random_state = 214243)
    pca.fit(X_train)
    X_train_pca500 = pca.transform(X_train)
    X_test_pca500 = pca.transform(X_test)
    
    pca = PCA(n_components=1000, random_state = 214243)
    pca.fit(X_train)
    X_train_pca1000 = pca.transform(X_train)
    X_test_pca1000 = pca.transform(X_test)


## Model Configuration
Brain Tumor 4가지를 분류하기 위해 KNN(K-Nearest Neighbors)과 SVM(Support Vector Machine)을 조합하여 사용함  
만약 KNN이 예측한 데이터가 'pituitary_tumor' 라고 한다면, 그것을 무조건 맞다고 가정하고 처리함  
만약 KNN이 예측한 데이터가 'pituitary_tumor' 가 아니라고 한다면 SVM을 사용하여 예측을 수행  

    knn = KNeighborsClassifier(n_neighbors=1)
    svc = SVC(C=12, gamma=8)
    knn.fit(X_train_pca500, y_train)
    svc.fit(X_train_pca1000, y_train)
    knn_pred = knn.predict(X_test_pca500)
    svc_pred = svc.predict(X_test_pca1000)
    y_pred = []
    for i in range(len(X_test_pca500)):
        if knn_pred[i] == labels[3]:
            y_pred.append(labels[3])
        else:
            y_pred.append(svc_pred[i])
    y_pred = np.array(y_pred)

## License
MIT License

## I am ...
E-mail : duekzin@naver.com  
ID : 20234584  
NAME : 최시우  
