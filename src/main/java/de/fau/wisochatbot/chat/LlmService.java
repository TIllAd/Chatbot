package de.fau.wisochatbot.chat;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.List;
import java.util.Map;

@Service
public class LlmService {

    private final WebClient client;
    private final String model;

    public LlmService(
            WebClient.Builder builder,
            @Value("${llm.base-url}") String baseUrl,
            @Value("${llm.model}") String model
    ) {
        this.client = builder.baseUrl(baseUrl).build();
        this.model = model;
    }

    public String reply(String message) {
        try {
            Map<String, Object> body = Map.of(
                    "model", model,
                    "messages", List.of(
                            Map.of("role", "system", "content", "Du bist der WiSo-Chatbot. Antworte kurz und hilfreich auf Deutsch."),
                            Map.of("role", "user", "content", message)
                    ),
                    "stream", false
            );

            Map<?, ?> res = client.post()
                    .uri("/api/chat")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (res == null) return "(keine Antwort)";

            // Ollama: res.message.content
            Object msgObj = res.get("message");
            if (msgObj instanceof Map<?, ?> msgMap) {
                Object content = msgMap.get("content");
                if (content != null) return content.toString();
            }

            // Fallback: manche Responses nutzen "response"
            Object response = res.get("response");
            if (response != null) return response.toString();

            return "(keine Antwort)";

        } catch (WebClientResponseException e) {
            return "⚠️ Ollama Fehler: HTTP " + e.getStatusCode().value() + " – " + e.getResponseBodyAsString();
        } catch (Exception e) {
            return "⚠️ Unerwarteter Fehler: " + e.getMessage();
        }
    }
    public String replyWithHistory(List<Map<String, String>> messages) {
        try {
            Map<String, Object> body = Map.of(
                    "model", model,
                    "messages", messages,
                    "stream", false
            );

            Map<?, ?> res = client.post()
                    .uri("/api/chat")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (res == null) return "(keine Antwort)";

            // Ollama: res.message.content
            Object msgObj = res.get("message");
            if (msgObj instanceof Map<?, ?> msgMap) {
                Object content = msgMap.get("content");
                if (content != null) return content.toString();
            }

            // Fallback: manche Responses nutzen "response"
            Object response = res.get("response");
            if (response != null) return response.toString();

            return "(keine Antwort)";

        } catch (WebClientResponseException e) {
            return "⚠️ Ollama Fehler: HTTP " + e.getStatusCode().value() + " – " + e.getResponseBodyAsString();
        } catch (Exception e) {
            return "⚠️ Unerwarteter Fehler: " + e.getMessage();
        }
    }
}