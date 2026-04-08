FROM ubuntu:22.04
RUN apt-get update
RUN apt-get -y install wget ssh curl openjdk-21-jdk python3-pip procps net-tools
RUN pip3 install jupyter pyspark

# Download and extract Spark
WORKDIR /opt/spark
RUN wget https://dlcdn.apache.org/spark/spark-4.1.1/spark-4.1.1-bin-hadoop3.tgz
RUN tar -zxvf spark-4.1.1-bin-hadoop3.tgz
RUN rm spark-4.1.1-bin-hadoop3.tgz

# Set Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV SPARK_HOME=/opt/spark/spark-4.1.1-bin-hadoop3
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PATH="$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"
ENV SPARK_NO_DAEMONIZE=true

# Expose ports
EXPOSE 8888
EXPOSE 8080
EXPOSE 7077
EXPOSE 4040

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--allow-root", "--ip=0.0.0.0"]