from flask import Flask, request, send_file
import speech_recognition as sr
import openai
import gtts
import os
import json
import cv2
import numpy as np
import ai

app = Flask(__name__)

recognizer = sr.Recognizer()

openai.api_key = "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"

@app.route('/process', methods=['POST'])
def process_text():
    try:
        
        recorded_audio = sr.AudioData(request.files['audio'].read(), sample_rate=44100, sample_width=2)
        
        text = recognizer.recognize_google(recorded_audio, language="tr-TR")
        print(text)
        
        if text == "":
            return 

        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Sen bir yapay zeka asistansın verdiğim soru ve/veya sorulara düzgün bir şekilde açıklayarak cevap vereceksin. Soru(lar):{text}",
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        generated_text = response["choices"][0]["text"]
        
        print(generated_text)

        tts = gtts.gTTS(text=generated_text, lang='tr')
        filename = "output.mp3"
        tts.save(filename)
        
        return filename
    
    except Exception as e:
        print(e)

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        image = request.files["image"]

        image = np.frombuffer(image.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        result = ai.recognize(image)

        print(result)

        tas_anlamlari = {
            "0": "Genç yaşta vefat eden hanım başlığı(Üzerinde güller ve bir kurdele bulunmakta.)",
            "cennetmeyvesi": "Meyveler(Bu mezar taşının üzerindeki meyveler bolluğu ve cennet meyvelerini simgeler.)",
            "denizci": "Gemi dümeni(Bu mezar taşı bir denizciyi sembolize etmekte. Denizci Mezar Taşları'ndaki önemli hususlardan biri genellikle mezar taşı üzerinde kişinin denizci olduğuna dair bir işaret, sembol veya motif bulunmasıdır. Bunlar gemi direği, yelken bezi, urgan, halat, gemi çapası, Osmanlı denizcilik arması gibi şekillerde karşımıza çıkmaktadır. Bunun yanı sıra kırık gemi direği tasvirleri, kabir sahibinin ölümü yani bu dünyadaki hayatının bitmiş olması ile ilgilidir. Bu arada bazı gemici lahitleri de vardır ki aynen gemiye benzetilmişlerdir. Bunların etraflarını boydan boya halatlar ve gemi zincirleri çevrelemektedir.)",
            "feslibaslik": "Fesli Başlık(Fesli mezar taşları Fesli başlıklar kişinin mesleği hakkında tam bir bilgi vermez. Sadece hangi dönemde yaşadıklarını anlarız. Fesli mezar taşlarının en büyük ve görkemlileri II. Mahmud döneminde kullanılan feslerdir.)",
            "gul": "Gül(Kadın mezar taşları Bir kadının incelik ve letafetini en güzel şekilde ortaya koyan şeyler, yani çiçekler, buketler, bahar dalları, gerdanlık, küpe ve broşlarla süslüdür.)",
            "gulveuzum": "Gül ve üzüm(Kadın mezar taşları Bir kadının incelik ve letafetini en güzel şekilde ortaya koyan şeyler, yani çiçekler, buketler, bahar dalları, gerdanlık, küpe ve broşlarla süslüdür.)",
            "hanimbaslik": "Hanım Başlık(Kadın mezar taşları Bir kadının incelik ve letafetini en güzel şekilde ortaya koyan şeyler, yani çiçekler, buketler, bahar dalları, gerdanlık, küpe ve broşlarla süslüdür.)",
            "hasanhilmiefendi": "Hasan Hilmi Efendi(Hasan Hilmi Efendi (1782-1847), Kıbrıs Türkü şair. 1782 yılında Kıbrıs'ta doğdu. Eğitimini Kıbrıs'ta aldı, Arapça, Farsça ve İslam alanlarında uzmanlaştı. Bir müddet bu alanlarda öğretmenlik yaptı. Daha sonra klasik Osmanlı stilinde, aruz vezninde şiirler yazmaya başladı. II. Mahmud ona 'şiirlerin sultanı' adını verdi. Ayrıca Kıbrıs müftüsü olarak atandı.)",
            "kafesidestarlikavuk": "Kafesi Destarlı Kavuk(Vergi, maliye ve hesap memurları başlığı.)",
            "katibikavuk": "Katibi kavuk(Devlet memurlarının giydiği başlık)",
            "mahmudifesli": "Mahmudi Fesli(Sultan 2. Mahmud dönemini simgeleyen başlık)",
            "meyvehanimbaslik": "Meyve içeren hanım başlık(Yatan kişinin bir hanım olduğunu ve bolluk içerisinde olduğunu simgelemektedir.)",
            "omerziyaeddinefendi": "Ömer Ziyaeddin Efendi(Ömer Ziyâeddin Efendi, gençlik yıllarında Kafkasya’da Ruslara karşı yıllardır sürdürülmekte olan mücadelelere katıldı. 1877-78 Osmanlı Rus Harbi’nde Osmanlı Devleti’nin savaşta yenilmesi neticesinde bölge tamamıyla Rus kontrolü altına girmiş, Dağıstan halkları ile birlikte Ömer Dağıstânî Hazretlerinin ailesi de Osmanlı topraklarına hicret etmiştir. Ulema ailesinden gelen ve Nakşi-Halidî tarikatıyla önceden bağı olan Ömer Ziyâeddin Efendi, payitaht İstanbul’a yerleştiklerinde Nakşibendî şeyhi Ahmed Ziyâeddin Gümüşhânevî Hazretlerine intisap eder. Ve böylelikle Dağıstan’da başladığı eğitimine İstanbul’da Gümüşhânevî Tekkesi6’nde devam eder. Çalışmalarındaki ciddiyeti, çalışkanlığı ve ihlası ile hocasının dikkatini çeken Dağıstânî Hazretleri’ne hocası “Oğlum, sana Ziyâeddin adını veriyorum, isminle muammer ol.” der ve bundan sonra ismi Ömer Ziyâeddin diye anılır.)",
            "orfidestarlikavuk": "Örfi destarlı kavuk(Örfi destarlı kavuk 2 Üst dereceden ulemaya aittir. Bu tür kavuklar, şeyhülislamlar, kazaskerler, üst dereceden kadılar, Mekke ve Medine'de hizmette bulunan hocalar takardı. Kallavi kavuk Osmanlı yönetiminde sadrazam, kubbealtı vezirleri ve kaptan-ı deryalar için kullanılırdı.)",
            "uzumvesedir": "Üzüm ve servi ağacı(Servi Ağacı Motifi; Yüce Allaha Tevekkül etmeyi ve Elif harfine benzemesiyle de yine Allah’ı simgelemektedir. Üzüm Salkımı Motifi; Bolluk ve bereketi simgeler.)"

        }

        print("BAŞARI RESİM")

        text = tas_anlamlari[result]
        print(text)
        
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=f"Sen bir yapay zeka asistanısın ve sana osmanlı mezar taşının üzerinde bulunan başlıklar veya motifler verilecek parantez içerisinde de bunun ne anlama geldiği belirtilecek bunu açıklayacaksın. Motif(ler):{text}",
            max_tokens=2048,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        generated_text = response["choices"][0]["text"]
        
        print(generated_text)

        tts = gtts.gTTS(text=generated_text, lang='tr')
        filename = "output.mp3"
        tts.save(filename)

        return json.dumps({'success':True}), 200, {'ContentType':'application/json'}
    except Exception as e:
        print(e)
        return(e)

@app.route('/download', methods=['GET'])
def download_file():
    try:
        filename = request.args.get('filename')
        return send_file(filename, as_attachment=True)
    except Exception as e:
        return print(e)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
