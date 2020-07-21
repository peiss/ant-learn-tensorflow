from flask import Flask, request
import numpy as np
import json
import tensorflow as tf

app = Flask(__name__)

# 加载模型文件
model = tf.keras.models.load_model("./heart_model.h5")


@app.route('/my_predict', methods=["POST", "GET"])
def index():
  result_html = ""
  if request.form.get("datajson"):
    # 如果有表单数据传入
    data = np.array(json.loads(request.form["datajson"]))
    result = str(model.predict(data))

    result_html = f"""
        <h2>输入内容：</h2>
        <pre>{data}</pre>

        <h2>预估结果：</h2>
        <pre>{result}</pre>
      """

  return f"""
    <html>
      <body>
        <center>
        <form action="/my_predict" method="post" style="font-size:18px">
          请输入JSON：<textarea name="datajson" rows="20" cols="50"> </textarea>
          <br /><br /><br />
          <input type="submit" />
        </form>
        
        <hr />
        {result_html}
        </center>
      </body>
    </html>
  """


if __name__ == '__main__':
  app.run()
