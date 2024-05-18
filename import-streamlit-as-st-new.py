import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from category_encoders import OrdinalEncoder

# Sayfa Ayarları
st.set_page_config(
    page_title="vehicle insurance fraud",
    menu_items={
        "Get help": "https://www.linkedin.com/in/azad-halhall%C4%B1-10b014284/",
        "About": "For More Information\n" + "https://github.com/azadhalhalli"
    }
)

# Model yolunu tanımlayın
model = joblib.load("C:/Users/halha/OneDrive/Belgeler/GitHub/proje3/gb_modelnew.pkl")

# Model dosyasının mevcut olup olmadığını kontrol edin
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")

    # Ordinal Encoder için tanımlanan eşlemeleri kullanarak ordinal encoding yapılacak sütunları tanımlayın
    col_ordering = [
        {'col':'PastNumberOfClaims','mapping':{'none':0 ,'1':1,'2 to 4':3,'more than 4':5 }},
        {'col':'NumberOfSuppliments','mapping':{'none':0,'1 to 2':1,'3 to 5':3,'more than 5':6}}, 
        {'col':'VehiclePrice','mapping':{'more than 69000':5,'20000 to 29000':1,'30000 to 39000':2,'less than 20000':0,
                                         '40000 to 59000':3,'60000 to 69000':4}},
        {'col':'AgeOfVehicle','mapping':{'new': 0, '2 years': 1, '3 years': 2, '4 years': 3, '5 years': 4, '6 years': 5, '7 years': 6, 'more than 7': 7}},
        {'col':'Year','mapping': {1994: 0, 1995: 1, 1996: 2}},
        {'col':'Days_Policy_Accident','mapping': {'none': 0, '1 to 7': 1,'8 to 15': 2,'15 to 30': 3, 'more than 30': 4}},
        {'col':'Days_Policy_Claim','mapping': {'none': 0, '1 to 7': 1,'8 to 15': 2,'15 to 30': 3, 'more than 30':4 }},
        {'col':'AddressChange_Claim','mapping': {'1 year': 1, 'no change': 0, '4 to 8 years': 4, '2 to 3 years': 2, 'under 6 months': 0.5}},
        {'col':'AgeCategory','mapping': {'genc': 0, 'orta': 1, 'yasli': 2}},
        {'col':'VehiclePrice_Cat','mapping': {'high': 2, 'mid': 1, 'low': 0}},
        {'col':'NumberOfCars','mapping': {'3 to 4': 3, '1 vehicle': 1, '2 vehicles': 2, '5 to 8': 7, 'more than 8': 9}},                  
    ]

    # Streamlit arayüz başlığı
    st.title("Car Insurance Fraud Detection")

    # Başlık Ekleme
    st.title("Machine Learning to Prevent Car Insurance Fraud: The New Weapon for Insurance Companies")

    # Markdown Oluşturma
    st.markdown("Car insurancefraud poses a serious threat to insurance companies. Such **fraud** not only raises insurance premiums but also undermines trust across the sector, **harming** both insurers and customers. However, technology is now stepping in to help tackle this problem.")

    # Resim Ekleme
    st.image("https://www.ginary.com.tr/wp-content/uploads/2022/06/makine-ogrenimi-ve-yapay-zeka.jpg")

    st.markdown("**Machine learning** has become a significant tool for insurance companies in detecting and preventing such fraudulent cases. **Machine learning** algorithms can analyze large amounts of data to identify patterns and trends indicative of fraud. By examining data from past insurance claims, these algorithms can be utilized to identify fraudulent cases.")
    st.markdown("A **machine learning** model used to detect car insurance fraud goes through several steps. First, a large dataset is collected to train the model. This dataset includes real insurance claims along with labels indicating whether they are fraudulent or not. Before training the model, the dataset is preprocessed and cleaned, missing data is filled, and categorical data is converted into numerical values.")

    st.image("https://storage.evrimagaci.org/old/content_media/cd394f612e40b92610eec164a1804712.png")

    st.markdown("Next comes the feature engineering step. In this step, new features potentially significant for detecting fraud are created. For example, features like the age of the car, the customer's history of insurance claims, and the claim amount are analyzed and utilized to enhance the model's accuracy.")

    st.markdown("During model training, various machine learning algorithms are tried, and the model that performs best is selected. These algorithms may include logistic regression, decision trees, random forests, and support vector machines. The selected model is then tested, and its performance is evaluated using various metrics such as accuracy, precision, recall, and F1 score.")
    st.markdown("In conclusion, the developed machine learning model provides insurance companies with a significant tool for detecting and preventing car insurance fraud. This model can help insurers reduce financial losses and provide better service to their customers. In the future, it is expected that these models will be further improved using more advanced machine learning techniques and larger datasets.")
    st.markdown("Therefore, machine learning plays a crucial role in the insurance sector and is considered a powerful tool in preventing car insurance fraud.")

    st.image("https://botekotomasyon.com/wp-content/uploads/2020/06/makine.jpg.webp")

    # Kullanıcıdan girdi alma
    st.header("Enter the details of the insurance claim")

    # Kullanıcıdan girdileri al
    Month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    WeekOfMonth = st.selectbox("Week of Month", [1, 2, 3, 4, 5])
    DayOfWeek = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    Make = st.text_input("Make (Car Manufacturer)")
    AccidentArea = st.selectbox    ("Accident Area", ["Urban", "Rural"])
    DayOfWeekClaimed = st.selectbox("Day of Week Claimed", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    MonthClaimed = st.selectbox("Month Claimed", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    WeekOfMonthClaimed = st.selectbox("Week of Month Claimed", [1, 2, 3, 4, 5])
    Sex = st.selectbox("Sex", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    Fault = st.selectbox("Fault", ["Policy Holder", "Third Party"])
    VehicleCategory = st.selectbox("Vehicle Category", ["Sedan", "SUV", "Sports Car", "Truck", "Van"])
    VehiclePrice = st.selectbox("Vehicle Price", ["more than 69000", "20000 to 29000", "30000 to 39000", "less than 20000", "40000 to 59000", "60000 to 69000"])
    RepNumber = st.number_input("Rep Number", min_value=0, step=1)
    Deductible = st.number_input("Deductible", min_value=0, step=1)
    DriverRating = st.slider("Driver Rating", 1, 5, 1)
    Days_Policy_Accident = st.selectbox("Days Policy (Accident)", ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"])
    Days_Policy_Claim = st.selectbox("Days Policy (Claim)", ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"])
    PastNumberOfClaims = st.selectbox("Past Number of Claims", ["none", "1", "2 to 4", "more than 4"])
    AgeOfVehicle = st.selectbox("Age of Vehicle", ["new", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "more than 7"])
    AgeOfPolicyHolder = st.number_input("Age of Policy Holder", min_value=0, step=1)
    PoliceReportFiled = st.selectbox("Police Report Filed", ["Yes", "No"])
    WitnessPresent = st.selectbox("Witness Present", ["Yes", "No"])
    AgentType = st.selectbox("Agent Type", ["External", "Internal"])
    NumberOfSuppliments = st.selectbox("Number of Supplements", ["none", "1 to 2", "3 to 5", "more than 5"])
    AddressChange_Claim = st.selectbox("Address Change (Claim)", ["1 year", "no change", "4 to 8 years", "2 to 3 years", "under 6 months"])
    NumberOfCars = st.selectbox("Number of Cars", ["3 to 4", "1 vehicle", "2 vehicles", "5 to 8", "more than 8"])
    Year = st.selectbox("Year", [1994, 1995, 1996])
    BasePolicy = st.selectbox("Base Policy", ["Liability", "Collision", "Comprehensive"])
    AgeCategory = st.selectbox("Age Category", ["genc", "orta", "yasli"])
    VehiclePrice_Cat = st.selectbox("Vehicle Price Category", ["high", "mid", "low"])

    # Submit button
    if st.button("Submit"):
        # Kullanıcı girdilerini bir DataFrame'e dönüştürme
        input_data = pd.DataFrame({
            'Month': [Month], 'WeekOfMonth': [WeekOfMonth], 'DayOfWeek': [DayOfWeek], 'Make': [Make],
            'AccidentArea': [AccidentArea], 'DayOfWeekClaimed': [DayOfWeekClaimed], 'MonthClaimed': [MonthClaimed],
            'WeekOfMonthClaimed': [WeekOfMonthClaimed], 'Sex': [Sex], 'MaritalStatus': [MaritalStatus],
            'Fault': [Fault], 'VehicleCategory': [VehicleCategory], 'VehiclePrice': [VehiclePrice],
            'RepNumber': [RepNumber], 'Deductible': [Deductible], 'DriverRating': [DriverRating],
            'Days_Policy_Accident': [Days_Policy_Accident], 'Days_Policy_Claim': [Days_Policy_Claim],
            'PastNumberOfClaims': [PastNumberOfClaims], 'AgeOfVehicle': [AgeOfVehicle],
            'AgeOfPolicyHolder': [AgeOfPolicyHolder], 'PoliceReportFiled': [PoliceReportFiled],
            'WitnessPresent': [WitnessPresent], 'AgentType': [AgentType], 'NumberOfSuppliments': [NumberOfSuppliments],
            'AddressChange_Claim': [AddressChange_Claim], 'NumberOfCars': [NumberOfCars], 'Year': [Year],
            'BasePolicy': [BasePolicy], 'AgeCategory': [AgeCategory], 'VehiclePrice_Cat': [VehiclePrice_Cat]
        })

        # Ordinal encoding
        ord_encoder = OrdinalEncoder(mapping=col_ordering, return_df=True)
        input_data = ord_encoder.fit_transform(input_data)

        # Kalan kategorik sütunları etiketleyin
        label_encoder = LabelEncoder()
        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = label_encoder.fit_transform(input_data[col])

        # Model ile tahmin yapın
        prediction = model.predict(input_data)

        # Sonucu göster
        if prediction[0] == 1:
            st.write("The claim is predicted to be fraudulent.")
        else:
            st.write("The claim is predicted to be genuine.")

