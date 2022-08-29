FROM python

# Install packages for image alpine 
# RUN apk update && apk add python3-dev gcc libc-dev

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python3", "main.py"]
