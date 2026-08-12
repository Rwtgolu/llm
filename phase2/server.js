import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import {RecursiveCharacterTextSplitter} from "@langchain/textsplitters"


dotenv.config();

const PORT = process.env.PORT;
const app = express();
app.use(express.json());


const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxTokens: 200,
  maxRetries: 2,
});


const upload=async ()=>{
    const pdfpath="./oops_interview.pdf";
    const buffer= fs.readFileSync(pdfpath)
    const pdfresult= await new PDFParse({data:buffer});
    const result= await pdfresult.getText();
    const textdata=result.text;
    const chunk= new RecursiveCharacterTextSplitter({
      chunkSize:500,
      chunkOverlap:100

    });
    const data=await chunk.createDocuments([textdata]);
    console.log(data);
}
upload();


app.post("/ai", async (req, res) => {
  try {
    const { input } = req.body;

    const response = await llm.invoke(input);




    res.status(200).json({
      message: response.content,
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({
      error: error.message,
    });
  }
});



app.get("/", (req, res) => {
  res.send("message hellow from server");
});

app.listen(PORT, () => {
  console.log(`server started at PORT ${PORT}`);
});