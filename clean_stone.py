import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.core.window import Window
from kivy.uix.popup import Popup

# Constants
TARGET_IMAGE_SIZE = (224, 224)
DROPOUT_RATE = 0.5
DENSE_LAYER_UNITS = 1024
FREEZE_LAYERS = 15
LEARNING_RATE = 0.0001
FONT_PATH = os.path.join("C:/Users/user/Desktop/수업자료/Project/kivy/kivy_apps/", "NanumGothic.ttf")
WINDOW_SIZE = (600, 600)
DEFAULT_DRIVES = ['C:/', 'D:/', 'E:/']
INITIAL_LABEL_TEXT = "이미지를 선택 후 예측 버튼을 클릭하세요."
RESULT_TEMPLATE = "결과: {prediction}\n확률: {confidence:.2f}"

# Fine-Tuning된 VGG16 모델 로드
def build_model():
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Fully connected 대신 평균 풀링 사용
    x = Dense(DENSE_LAYER_UNITS, activation='relu')(x)  # 큰 Dense Layer 추가
    x = Dropout(DROPOUT_RATE)(x)  # Dropout으로 과적합 방지
    predictions = Dense(2, activation='softmax')(x)  # Cracked / Uncracked

    model = Model(inputs=base_model.input, outputs=predictions)

    # 기본 모델의 가중치를 동결 (첫 몇 개 층을 학습되지 않게 설정)
    for layer in base_model.layers[:FREEZE_LAYERS]:
        layer.trainable = False

    # Adam optimizer로 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 로드
model = build_model()

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=TARGET_IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# 예측 함수
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    preds = model.predict(img_array)
    cracked_confidence = preds[0][0]
    uncracked_confidence = preds[0][1]
    confidence = max(cracked_confidence, uncracked_confidence)
    class_name = "cracked" if cracked_confidence > uncracked_confidence else "uncracked"
    return class_name, confidence

# 버튼 생성 함수
def create_button(text, callback, size_hint=(0.4, 1)):
    button = Button(text=text, size_hint=size_hint, font_name=FONT_PATH, font_size=18)
    button.bind(on_release=callback)
    return button

# Kivy 앱 클래스
class CrackDetectionApp(App):
    def build(self):
        Window.size = WINDOW_SIZE

        # 메인 레이아웃
        self.main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # 이미지 표시 위젯
        self.image_widget = Image(source='', size_hint=(1, 0.6))
        self.image_widget.bind(on_touch_down=self.show_file_chooser)
        self.main_layout.add_widget(self.image_widget)

        # 예측 결과 출력 레이블
        self.result_label = Label(text=INITIAL_LABEL_TEXT, size_hint=(1, 0.1),
                                  font_name=FONT_PATH, font_size=18, color=(1, 1, 1, 1))
        self.main_layout.add_widget(self.result_label)

        # 버튼 레이아웃
        self.button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)

        # 버튼 추가
        self.button_layout.add_widget(create_button("예측하기", self.predict_image_class))
        self.button_layout.add_widget(create_button("처음으로", self.reset_app))
        self.button_layout.add_widget(create_button("종료", self.stop_app))

        self.main_layout.add_widget(self.button_layout)
        return self.main_layout

    def show_file_chooser(self, instance, touch):
        if instance.collide_point(*touch.pos):
            filechooser_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
            filechooser = FileChooserIconView(path='C:/', size_hint=(1, 0.7))

            # 드라이브 선택 드롭다운 메뉴
            dropdown = DropDown()
            for drive in DEFAULT_DRIVES:
                btn = Button(text=drive, size_hint_y=None, height=30, font_name=FONT_PATH)
                btn.bind(on_release=lambda btn: self.update_filechooser_path(filechooser, btn.text))
                dropdown.add_widget(btn)

            drive_button = Button(text='드라이브 선택', size_hint_y=None, height=40, font_name=FONT_PATH)
            drive_button.bind(on_release=dropdown.open)

            path_input = TextInput(hint_text='경로를 입력하세요', size_hint_y=None, height=40, multiline=False,
                                    font_name=FONT_PATH, font_size=18)
            path_input.bind(on_text_validate=lambda instance: self.update_filechooser_path(filechooser, instance.text))

            filechooser_layout.add_widget(drive_button)
            filechooser_layout.add_widget(path_input)
            filechooser_layout.add_widget(filechooser)

            popup = Popup(title="이미지 선택", content=filechooser_layout, size_hint=(0.9, 0.9), title_font=FONT_PATH)

            def select_image(filechooser_instance, selection, touch=None):
                if selection:
                    selected_path = os.path.abspath(selection[0])
                    self.image_widget.source = selected_path
                    self.image_path = selected_path
                    self.result_label.text = "이미지를 선택했습니다."
                    popup.dismiss()

            filechooser.bind(on_submit=select_image)
            popup.open()

    def update_filechooser_path(self, filechooser, path):
        if os.path.exists(path):
            filechooser.path = path
        else:
            self.show_error_popup("입력하신 경로가 존재하지 않습니다.")

    def show_error_popup(self, message):
        popup_content = BoxLayout(orientation='vertical')
        popup_content.add_widget(Label(text=message, font_name=FONT_PATH))
        close_button = create_button("확인", lambda instance: popup.dismiss(), size_hint=(1, 0.3))
        popup_content.add_widget(close_button)
        popup = Popup(title="경로 오류", content=popup_content, size_hint=(0.5, 0.3), title_font=FONT_PATH)
        popup.open()

    def predict_image_class(self, instance):
        if hasattr(self, 'image_path') and self.image_path:
            prediction, confidence = predict_image(self.image_path)
            self.result_label.text = RESULT_TEMPLATE.format(prediction=prediction, confidence=confidence)
        else:
            self.result_label.text = "이미지를 먼저 선택해주세요."

    def reset_app(self, instance):
        self.image_widget.source = ''
        self.image_path = None
        self.result_label.text = INITIAL_LABEL_TEXT

    def stop_app(self, instance):
        self.stop()

if __name__ == "__main__":
    CrackDetectionApp().run()
