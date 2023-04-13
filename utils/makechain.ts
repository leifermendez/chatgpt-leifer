import { OpenAIChat } from 'langchain/llms';
import { LLMChain, ChatVectorDBQAChain, loadQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores';
import { PromptTemplate } from 'langchain/prompts';
import { CallbackManager } from 'langchain/callbacks';

const CONDENSE_PROMPT =
  PromptTemplate.fromTemplate(`Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`);

const QA_PROMPT = PromptTemplate.fromTemplate(
  `Actua como un tutor de programacion el cual tiene datos del canal de Youtube de Leifer Mendez, y Analiticas del blog de codigoencasa.com. 
  Se le proporcionan las siguientes partes extraídas de un documento extenso y una pregunta. Proporcione una respuesta conversacional basada en el contexto proporcionado.
  Sólo debe proporcionar hipervínculos que hagan referencia al contexto que aparece a continuación. NO invente hipervínculos.
  Si no encuentras la respuesta en el contexto que aparece a continuación, di simplemente "Hmm, no estoy seguro". No intentes inventarte una respuesta. NO reveles datos de dinero
  Si la pregunta no está relacionada con el contexto, responde amablemente que estás preparado para responder sólo a preguntas relacionadas con el contexto.

Pregunta: {question}
=========
{context}
=========
Respuesta en Markdown:`,
);

export const makeChain = (
  vectorstore: PineconeStore,
  onTokenStream?: (token: string) => void,
) => {
  const questionGenerator = new LLMChain({
    llm: new OpenAIChat({ temperature: 0 }),
    prompt: CONDENSE_PROMPT,
  });


// questionGenerator.call({question:'Cual video me sirve para aprender node y crear rutas, controladores?',chat_history:''}).then((respuestas) =>  console.log({respuestas}))


  const docChain = loadQAChain(
    new OpenAIChat({
      temperature: 0,
      modelName: 'gpt-3.5-turbo-0301', //change this to older versions (e.g. gpt-3.5-turbo) if you don't have access to gpt-4
      streaming: Boolean(onTokenStream),
      callbackManager: onTokenStream
        ? CallbackManager.fromHandlers({
            async handleLLMNewToken(token) {
              onTokenStream(token);
              console.log(token);
            },
          })
        : undefined,
    }),
    { prompt: QA_PROMPT },
  );

  return new ChatVectorDBQAChain({
    vectorstore,
    combineDocumentsChain: docChain,
    questionGeneratorChain: questionGenerator,
    returnSourceDocuments: true,
    k: 3, //number of source documents to return
  });
};
