import express from "express";
import dotenv from "dotenv";
import { ChatGroq } from "@langchain/groq";
import fs from "fs";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";

import { QdrantVectorStore } from "@langchain/qdrant";

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

const embeddings = new GoogleGenerativeAIEmbeddings({
  model: "gemini-embedding-001", // 768 dimensions
  taskType: "RETRIEVAL_DOCUMENT",
  title: "Document title",
});

const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, {
  url: process.env.QDRANT_URL,
  apiKey: process.env.QDRANT_API_KEY,
  collectionName: "gaurav",
});

const upload = async () => {
  const pdfpath = "./oops_interview.pdf";
  const buffer = fs.readFileSync(pdfpath);
  const pdfresult = await new PDFParse({ data: buffer });
  const result = await pdfresult.getText();
  const textdata = result.text;
  const chunk = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 100,
  });
  const data = await chunk.createDocuments([textdata]);
  await vectorStore.addDocuments(data);
  console.log(data);
};

app.post("/ai", async (req, res) => {
  try {
    const { input } = req.body;

    if (!input) {
      return res.status(400).json({ error: "Input is required." });
    }

    const docs = await vectorStore.similaritySearch(input, 5);

    const context = docs.map((doc) => doc.pageContent).join("\n\n");

    const response = await llm.invoke([
      {
        role: "system",
        content: `You are a RAG-based AI assistant.
Answer the user's question using the provided context.
If the answer is not available in the context, say that you don't know.

Context:
${context}`,
      },
      {
        role: "human",
        content: input,
      },
    ]);

    // 4. Send the response back to the client
    return res.json({ answer: response.content });
  } catch (error) {
    console.error("Error processing AI request:", error);
    return res.status(500).json({ error: "Internal server error." });
  }
});

app.get("/", (req, res) => {
  res.send("message hellow from server");
});

app.listen(PORT, () => {
  console.log(`server started at PORT ${PORT}`);
});
