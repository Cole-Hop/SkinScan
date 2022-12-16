//SkinScan (Capstone CSU Chico, Fall 2022)
//Developed by Cole Hopkins
//Adviser: Sam Siewert

//This app was developed to allow for the model I developed to be deployed and usable by users to
//get fast and easy results based off of the image they use

import React, {useState, useEffect, setState, useRef} from 'react'; //react hooks and data management
import { StatusBar } from 'expo-status-bar'; //change color for status bar icons
import { Image, StyleSheet, Button, Modal, Text, View, ScrollView, TouchableOpacity, Linking, LogBox} from 'react-native'; //react native render views
//import Modal from "react-native-modal";
import Renew from 'react-native-restart';
import { render } from 'react-dom';
import * as ImagePicker from 'expo-image-picker'; //for picking images
import {Camera, CameraType, WhiteBalance } from 'expo-camera'; //for using camera

import AnimatedLoader from "react-native-animated-loader"; //loader on app startup
import {bundleResourceIO, decodeJpeg} from '@tensorflow/tfjs-react-native' //tools for bundling pox model info
import * as tf from '@tensorflow/tfjs' //tf package for model access
import { reduceEachLeadingCommentRange } from 'typescript';
import { IFFT } from '@tensorflow/tfjs';
import { NativeModules } from "react-native";

const modelJSON = require("./react-model/model.json") //load in model information
const modelWeights = require('./react-model/group1-shard.bin') //load in weights of model

LogBox.ignoreAllLogs();//Ignore all log notifications

export default function App()
{
  const [isTfReady, checkTJ] = useState(false); //tensorflow package properly installed on startup
  const [load, setLoad] = useState(true); //unused
  const [isModelReady, checkMD] = React.useState(false); //loaded model correctly on startup
  const [image, setImage] = useState(null); //store image data from photo
  const [permission, requestPermission] = Camera.useCameraPermissions(); //get permission for camera access
  const [camera, setCamera] = React.useState(null); //camera being used
  const [pred, setPred] = useState(null); //get pred values for display
  const [type, setType] = useState(CameraType.back); //switch camera to front facing
  const [isModalVisible, setModalVisible] = useState(true);
  const cameraRef = useRef(null);

  useEffect(() => {
    async function appStartup()
    {
      if(!permission) //camera permissions are loading
      {
        console.log("Loading Camera Permissions")
        return <View />
      }
      if(!permission.granted) //access to camera has not been granted yet
      {
        console.log("Permissions not granted for camera use, requesting...")
        return(
          <View style={styles.permissions}>
            <Text>Allow SkinScan to access your camera?</Text>
            <Button onPress={requestPermission} title="allow access" /> 
          </View>
        )
        //ask for user permissions. If user accepts, load into app
      }
      
      console.log("Application started")
      await tf.setBackend('cpu');
      console.log("Setting model to use cpu")
      console.log(tf.getBackend());
      tf.ready().catch((e)=>{
        console.log("error loading tf:", e)
      })
      checkTJ('true')
      console.log("tf ready")
      tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights)).catch((e)=>{
        console.log("error loading model:",e)
      })
      checkMD('true')
      console.log("model ready")
    }
    appStartup();
  }, [])

  setTimeout(() => {
    setLoad(false)
  }, 1000)

  const toggleModal = () => {
    const [isModalVisible, setModalVisible] = useState(false);
  };


  const loadModel = async()=>
  {
    console.log("oh no")
    //load in model from poxscan model data imported into app for use of predicting images
    const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights))
    return model
  }

  const getPredictions = async (image)=>
  {
    const model = await loadModel() //make sure model is loaded and ready to be used
    const tensor_image = await transformImageToTensor(image) //convert image to tensor to feed into model
    const predictions = await makePredictions(1, model, tensor_image) //predict image from model with 1 output

    const prediction = predictions[0] //get highest confidence image
    //setPred(prediction) 
    return prediction //return the prediction
  }

  const transformImageToTensor = async (uri)=>
  {
    //convert image to base64 
    const img64 = await FileSystem.readAsStringAsync(uri, {encoding:FileSystem.EncodingType.Base64})
    
    //buffer image
    const imgBuffer =  tf.util.encodeString(img64, 'base64').buffer

    //convert to raw version
    const raw = new Uint8Array(imgBuffer)

    //decode jpeg
    let imgTensor = decodeJpeg(raw)

    //resize tensor to 128x128, the same size used to train and validate the model
    imgTensor = tf.image.resizeNearestNeighbor(imgTensor, [128, 128])

    //normalization variable
    const scalar = tf.scalar(255)

    //normalize tensor image
    const tensorScaled = imgTensor.div(scalar)

    //reshape tensor image to load into model
    const img = tf.reshape(tensorScaled, [1,128,128,3])
    return img
  }

  const makePredictions = async ( batch, model, imagesTensor )=>
  {
    //predict tensor image
    const predictionsdata = model.predict(imagesTensor)

    //split by one
    let pred = predictionsdata.split(batch)
    return pred
  }

  //allow taking picture, and store it in data
  const takePicture = async () => 
  {
    //check that permissions have been allowed, if not don't access camera
    const {status} = await Camera.getCameraPermissionsAsync()
    if(status === 'granted' && cameraRef) //access camera
    {
      const data = await camera.takePictureAsync(null) //store picture data
      setImage(data.uri) //store image in function
    }
  }

  //allow selecting image from camera gallery, and store it in data
  const pickImage = async () =>  
  {
    //open gallery of 1:1 image, and store data in result
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.All,
      allowsEditing: true,
      aspect: [3, 3],
      quality: 1,
    });

    console.log(result); //display result in log

    if (!result.cancelled) 
    {
      setImage(result.uri); //store image in function
    }
  };

  return(
    <View style={{ flex: 1, alignItems: 'center'}}>
      <StatusBar style="light"/> 
      <Text style={styles.header}></Text>
      <Text style={styles.title}>SkinScan</Text> 
      <AnimatedLoader
        visible={load}
        overlayColor="rgba(255,255,255,0)"
        source={require("./assets/loaders/loadplane.json")}
        animationStyle={styles.lottie}
        speed={0.5}>
      </AnimatedLoader> 
      <View style={styles.cameraContainer}>
        <Camera 
        ref={ref => setCamera(ref)}
        style={styles.fixedRatio} 
        type={type}
        ratio={'1:1'} />
      </View>
      <ScrollView style={styles.scrollView}>
        <Text> TensorFlow Ready:  
          {isTfReady ? <Text>Yes</Text> : ''}
        </Text>
        <Text> PoxScan Model Ready: 
          {isModelReady ? <Text>Yes</Text> : ''}
        </Text>
        <Text style={styles.titleText}>
        {"\n"}Welcome!</Text><Text style={styles.text}>
        {"\n"}SkinScan lets you take images of your skin, and get accurate results on what that skin condition
        may be. Currently, the model can detect...{"\n\n"}<Text style={styles.italicText}>(Chickenpox, Measels, Shingles)</Text>{"\n\n"}Future plans
        are to get the app to detect the following...{"\n\n"}<Text style={styles.italicText}>(Monkeypox, Acne, Mosquito Bites)</Text>{"\n\n"}
        <Text style={styles.boldText}>How to use the app</Text>{"\n"}1) Choose to either take a photo,
        or choose and image from your photo library {"\n"}2) Get results, it's just that easy!{"\n\n"}<Text style={styles.boldText}>Benefits
        to using SkinScan</Text>{"\n"}We do not collect the data of our users, and respect their privacy as we know these skin conditions can often
        be in revealing areas. The model used to predict your images is install with the app, and runs only locally on your device. I wanted to develop
        an app that also gives you speedy results when time is of the essence. A good focus was also put into focusing of getting a wide range of different
        skin pigmentations and lighting to feed the model, so that nobody is left out.{"\n\n"}<Text style={styles.boldText}>Backstory</Text>{"\n"}I first
        thought of the idea to make a model after taking a Machine Learning class, and seeing the benefit it can have on the world, espeically in medical use.
        During this time, MonkeyPox was also becoming a big deal. I got the idea to use this idea to broaden my understanding of Machine Learning, while also
        making a product that can have significance to peoples lives.{"\n\n"}<Text style={styles.boldText}>More Info</Text>{"\n"}If I was to take this project
        past my capstone goals that I set for myself, I would probably implement the following{"\n\n"}1) Allow for users to submit their image with their consent,
        allowing for more images to potentially retrain the model and get even more accurate predictions{"\n"}2) Create more classes of skin diseases that the
        model can accurately predict. Unfortunately, I had tried to get MonkeyPox to work while developing the model, but the available images online were sparse.
        {"\n"}3) Give more information to the user about the disease which the model predicts the user has a high chance of having (~%80 confidence){"\n"}4) Add an
        account option, to allow users to keep a history of their results of images.{"\n\n"}Thank you so much for looking at my app, if you have any questions or
        wish to reach out to me, check out my LinkedIn!{"\n\n"}linkedin.com/in/colehopkins
        </Text>
      </ScrollView>
      <View style={styles.row}>
          <View style={styles.buttonCam}>
              <Button color="#ff5c5c" title="Take Photo" onPress={takePicture} /> 
              {image && <Image source={{ uri: image }} style={{ width: 200, height: 200 }} />}
          </View>
          <View style={styles.buttonCam}>
              <Button color="#ff5c5c" title="Camera Roll" onPress={pickImage} /> 
              {image && <Modal isVisible={isModalVisible}>
                <View style={{ flex: 1, alignItems: "center"}}>
                  <Text style={styles.titleText}>{"\n"}Results</Text>
                  <Image source={{ uri: image }} style={{ width: 350, height: 350, justifyContent: "center", alignItems: "center" }} />
                  <Text style={styles.text}> chickenpox: </Text>
                  <Text style={styles.text}> shingles: </Text>
                  <Text style={styles.text}> measels: </Text>
                  <Button title="back" onPress={pickImage} />
                  <Text style={styles.text}> </Text>
                </View>
              </Modal>}
          </View>
      </View>
    </View>
  );
};

//{makePredictions(image)}

const styles = StyleSheet.create(
{
  modal:
  {
    flex: 1, 
    justifyContent: "center", 
    alignItems: "center"
  },

  scrollView: 
  {
    marginHorizontal: 5,
    flex: 1,
    backgroundColor: "#e8edfc",
  },

  boldText: 
  {
    fontWeight: 'bold',
  },

  italicText: 
  {
    fontStyle: 'italic',
  },

  text: 
  {
    textAlign: 'left',
    fontSize: 20,
  },

  titleText: 
  {
    textAlign: 'center',
    fontSize: 26,
  },

  buttonCam: 
  {
    flex: 1,
    justifyContent: 'flex-end',
    marginBottom: 15,
    paddingHorizontal: 10,
  },

  row: 
  {
    flex: .1,
    backgroundColor: '#127bb8',
    flexDirection: "row",
    flexWrap: "wrap",
  },

  header: 
  {
    backgroundColor: '#127bb8',
    flex: .13,
    paddingHorizontal: 250,
  },

  title: 
  {
    justifyContent: 'flex-end',
    marginTop: -60,
    fontSize: 40,
    color: "#ffffff",
    marginBottom: 7
  },

  permissions:
  {
    flex: 1,
    justifyContent: 'center',
  }
})

