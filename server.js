import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq"
import { context } from "@langchain/core/utils/context";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

dotenv.config();

const PORT=process.env.PORT;
const app=express();

app.use(express.json());

// const tools=[];
// const toolNode=new ToolNode(tools);

const llm = new ChatGroq({
   apiKey: process.env.GROQ_API_KEY,
    model: "openai/gpt-oss-120b",
    temperature: 0,
    maxTokens:200,
    maxRetries: 2,

})

const state=Annotation.Root({
    prompt:Annotation,
    aimsg:Annotation

})

const callLLM=async (state)=>{
    console.log(state)
     const response=await llm.invoke([
        {
            role:"system",
            content:"if you dont know then use relavent tools "
        },
        {
            role:"human",
            content:state.prompt

        }
     ])
    
     return {aimsg:response.content};

}

const graph=new StateGraph(state)
.addNode("agent",callLLM)
.addNode("tools",ToolNode)
.addEdge("__start__","agent")
.addEdge("agent","__end__")
.compile()






app.post("/ai",async(req,res)=>{
    const {input}=req.body;
     const response=await graph.invoke({prompt:input});
      console.log(response)
    res.status(200).json({"message":response.aimsg});

})

app.get("/",(req,res)=>{
    res.send("message hellow from server");
})

app.listen(PORT,()=>{
    console.log(`server started at PORT ${PORT}`)
})