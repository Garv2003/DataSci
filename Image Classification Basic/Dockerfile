# FROM tensorflow/tensorflow:2.6.0
# EXPOSE 5000
# WORKDIR /app
# COPY ./requirements.txt requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt
# COPY . .
# CMD ["flask", "run", "--host", "0.0.0.0"]


# Use an official TensorFlow runtime with Python 3.8
FROM tensorflow/tensorflow:2.6.0

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
