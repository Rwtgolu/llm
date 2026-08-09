import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq"
import { context } from "@langchain/core/utils/context";
dotenv.config();

const PORT=process.env.PORT;
const app=express();

app.use(express.json());

const llm = new ChatGroq({
   apiKey: process.env.GROQ_API_KEY,
    model: "openai/gpt-oss-120b",
    temperature: 0,
    maxTokens:200,
    maxRetries: 2,

})

app.post("/ai",async(req,res)=>{
    const {input}=req.body;
    const response=await llm.invoke(input)
    res.status(200).json({"message":response.content});

})

app.get("/",(req,res)=>{
    res.send("message hellow from server");
})

app.listen(PORT,()=>{
    console.log(`server started at PORT ${PORT}`)
})