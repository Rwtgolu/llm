import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import { TavilySearch } from "@langchain/tavily";
import {
  MessagesAnnotation,
  StateGraph,
  START,
  END,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

dotenv.config();

const PORT = process.env.PORT;
const app = express();
app.use(express.json());


const browserSearch = new TavilySearch({
  maxResults: 5,
  topic: "general",
  searchDepth: "basic",
});



const tools = [browserSearch];
const toolNode = new ToolNode(tools);



const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-120b",
  temperature: 0,
  maxTokens: 200,
  maxRetries: 2,
});




const llmWithTools = llm.bindTools(tools);

const callLLM = async (state) => {
  console.log(state);

  const response = await llmWithTools.invoke([
    {
      role: "system",
      content:
        "if you dont know anythink dont give incorrect ans"
    },
    ...state.messages,
  ]);

  return {
    messages: [response],
  };
};



const shouldContinue = (state) => {
  const lastMessage = state.messages[state.messages.length - 1];

  if (lastMessage.tool_calls?.length > 0) {
    return "tools";
  }

  return END;
};

const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callLLM)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent")
  .compile();




app.post("/ai", async (req, res) => {
  try {
    const { input } = req.body;

    const response = await graph.invoke({
      messages: [
        {
          role: "user",
          content: input,
        },
      ],
    });




    const lastMessage =
      response.messages[response.messages.length - 1];

    res.status(200).json({
      message: lastMessage.content,
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