[TOC]

# Installation and Deployment Instructions


## Hadoop environment configuration instructions

### Hadoop install

Execute the following command to unzip the downloaded JDK1.8 installation package.

```shell
tar -zxvf openjdk-8u41-b04-linux-x64-14_jan_2020.tar.gz
```

Execute the following command to move and rename the JDK package

```shell
mv java-se-8u41-ri/ /usr/java8
```

Execute the following command to configure Java environment variables

```shell
echo 'export JAVA_HOME=/usr/java8' >> /etc/profile
echo 'export PATH=$PATH:$JAVA_HOME/bin' >> /etc/profile
source /etc/profile
```

Execute the following command to check if Java has been successfully installed

```shell
java -version
```

download Hadoop

```shell
wget --no-check-certificate https://mirrors.bfsu.edu.cn/apache/hadoop/common/hadoop-2.10.1/hadoop-2.10.1.tar.gz
```

```shell
tar -zxvf hadoop-2.10.1.tar.gz -C /opt/
mv /opt/hadoop-2.10.1 /opt/hadoop
```

set Hadoop environment variables

```shell
echo 'export HADOOP_HOME=/opt/hadoop/' >> /etc/profile
echo 'export PATH=$PATH:$HADOOP_HOME/bin' >> /etc/profile
echo 'export PATH=$PATH:$HADOOP_HOME/sbin' >> /etc/profile
source /etc/profile
```

set yarn-env.sh and hadoop-env.sh

```shell
echo "export JAVA_HOME=/usr/java8" >> /opt/hadoop/etc/hadoop/yarn-env.sh
echo "export JAVA_HOME=/usr/java8" >> /opt/hadoop/etc/hadoop/hadoop-env.sh
```

try and test

```shell
hadoop version
```



### Hive install

#### Hive path

1）Hive path
http://hive.apache.org/
2）document
https://cwiki.apache.org/confluence/display/Hive/GettingStarted
3）download path
http://archive.apache.org/dist/hive/
4）github path
https://github.com/apache/hive

#### step

unzip to /usr/local

```shell
sudo tar -zxvf ./apache-hive-3.1.2-bin.tar.gz -C /usr/local
cd /usr/local/
```

rename the folder to hive

```shell
sudo mv apache-hive-3.1.2-bin hive 
```

set Hive environment variables

```shell
sudo vim /etc/profile
```

add to the first line

```shell
export HIVE_HOME=/usr/local/hive
export PATH=$PATH:$HIVE_HOME/bin
```

save and quit

```shell
source /etc/profile
```

exchange hive-site.xml  in `/usr/local/hive/conf`

```shell
#进入到hive的配置目录
cd /usr/local/hive/conf
#然后我们复制一个模板
sudo cp hive-default.xml.template hive-site.xml
```

```shell
sudo vim hive-site.xml
```

change hive-site.xml：

```xml
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
        <property>
                <name>javax.jdo.option.ConnectionURL</name>
                <value>jdbc:mysql://localhost:3306/hive</value>（mysql地址localhost）
        </property>
        <property>
                <name>javax.jdo.option.ConnectionDriverName</name>（mysql的驱动）
                <value>com.mysql.cj.jdbc.Driver</value>
        </property>
        <property>SS
                <name>javax.jdo.option.ConnectionUserName</name>（用户名）
                <value>root</value>
        </property>

        <property>
                <name>javax.jdo.option.ConnectionPassword</name>（密码）
                <value>123456</value>
        </property>
        <property>
                <name>hive.metastore.schema.verification</name>
                <value>false</value>
        </property>
</configuration>
```

do this

```shell
sudo cp mysql-connector-java-8.0.22.jar /usr/local/hive/lib/
```

initialization

```shell
schematool -dbType mysql -initSchema
```

#### start

start Hadoop

```shell
start-all.sh
```

start hive

```shell
cd /usr/local/hive/bin
hive
```



#### Sqoop install

下载Sqoop安装包：

下载地址（清华源）：https://mirrors.tuna.tsinghua.edu.cn/apache/sqoop/

上传并解压放到 /usr/local 文件夹下：

```shell
tar -zxvf sqoop-1.4.7.bin__hadoop-2.6.0.tar.gz -C /usr/local/
```

修改配置目录中的文件sqoop-env.sh

到指定目录下：/usr/local/sqoop-1.4.7.bin__hadoop-2.6.0/conf

重命名配置文件

```shell
mv sqoop-env-template.sh sqoop-env.sh
```

配置环境变量：

```shell
vim sqoop-env.sh

export HADOOP_COMMON_HOME=/usr/local/hadoop-2.7.7
export HADOOP_MAPRED_HOME=/usr/local/hadoop-2.7.7
export HIVE_HOME=/usr/local/apache-hive-1.2.2-bin
```

将mysql的驱动包mysql-connector-java-5.1.46-bin.jar复制到Sqoop安装目录下的lib文件夹中

配置环境变量：

```shell
vim ~/.bashrc

#sqoop
export SQOOP_HOME=/usr/local/sqoop-1.4.7.bin__hadoop-2.6.0
export PATH=$PATH:$SQOOP_HOME/bin

source ~/.bashrc

```

查看是否配置成功及版本号

```shell
sqoop version
```



### Scala

###### pom.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.FinalExam</groupId>
    <artifactId>url_DB</artifactId>
    <version>1.0-SNAPSHOT</version>

    <properties>
        <maven.compiler.source>8</maven.compiler.source>
        <maven.compiler.target>8</maven.compiler.target>
        <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    </properties>

    <dependencies>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-core_2.12</artifactId>
            <version>3.2.3</version>
        </dependency>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-sql_2.12</artifactId>
            <version>3.2.3</version>
        </dependency>
        <dependency>
            <groupId>mysql</groupId>
            <artifactId>mysql-connector-java</artifactId>
            <version>8.0.33</version>
        </dependency>

        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-hive_2.12</artifactId>
            <version>3.2.3</version>
            <!--<scope>provided</scope>-->
        </dependency>

    </dependencies>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-assembly-plugin</artifactId>
                <version>2.5.5</version>
                <configuration>
                    <archive>
                        <manifest>
                            <mainClass>org.FinalExam</mainClass>
                        </manifest>
                    </archive>
                    <descriptorRefs>
                        <descriptorRef>jar-with-dependencies</descriptorRef>
                    </descriptorRefs>
                </configuration>
                <executions>
                    <execution>
                        <id>make-assembly</id>
                        <phase>package</phase>
                        <goals>
                            <goal>single</goal>
                        </goals>
                    </execution>
                </executions>
            </plugin><plugin>
            <groupId>org.scala-tools</groupId>
            <artifactId>maven-scala-plugin</artifactId>
            <version>2.15.2</version>
            <executions>
                <execution>
                    <goals>
                        <goal>compile</goal>
                        <goal>testCompile</goal>
                    </goals>
                </execution>
            </executions>
        </plugin>
        </plugins>
    </build>

</project>
```

###### 打包

1. 在IDEA右侧点开maven，运行周期->package得到jar包

2. 在target中找到那个有dependencies的，复制到/opt/spark/bin下然后改名，执行命令

   ```shell
   sudo ./spark-submit --class org.FinalExam.要执行的类  --master spark://localhost:7077 ./你改的名.jar
   ```



