# Ubuntu 22.04 기반 이미지 사용
FROM ubuntu:22.04

# 패키지 목록 업데이트 (apt 패키지 최신 목록 다운로드)
RUN apt-get update

# wget(파일 다운로드 도구)와 Java 21 JDK 설치
# Spark는 Java 기반이라 JDK가 반드시 필요
RUN apt-get -y install wget openjdk-21-jdk

# 작업 디렉토리를 /opt/spark 로 설정
# 이후 RUN, CMD 등 명령은 이 경로에서 실행됨
WORKDIR /opt/spark

# Apache Spark 압축 파일 다운로드
# Spark 4.1.1 + Hadoop3 빌드 버전
RUN wget https://dlcdn.apache.org/spark/spark-4.1.1/spark-4.1.1-bin-hadoop3.tgz

# 다운로드한 Spark 압축 파일(.tgz) 압축 해제
RUN tar -zxvf spark-4.1.1-bin-hadoop3.tgz

# 압축 해제 후 필요 없는 tgz 파일 삭제 (이미지 용량 감소)
RUN rm spark-4.1.1-bin-hadoop3.tgz

# Java 설치 경로 환경변수 설정
# Spark가 Java runtime을 찾을 때 사용
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# Spark 설치 경로 환경변수 설정
# Spark 관련 명령어에서 기본 경로로 사용
ENV SPARK_HOME=/opt/spark/spark-4.1.1-bin-hadoop3

# PATH 환경변수에 Spark 실행 파일 경로 추가
# spark-shell, spark-submit 등의 명령을 어디서든 실행 가능
ENV PATH="$PATH:${SPARK_HOME}/bin:${SPARK_HOME}/sbin"

# Spark를 daemon(백그라운드 서비스)로 실행하지 않도록 설정
# Docker에서는 프로세스가 foreground로 실행되어야 컨테이너가 종료되지 않음
ENV SPARK_NO_DAEMONIZE=true