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
            @Value("${llm.model}") String model) {
        this.client = builder.baseUrl(baseUrl).build();
        this.model = model;
    }

    public String reply(String message) {
        try {
            Map<String, Object> body = Map.of("message", message);

            Map<?, ?> res =
                client
                    .post()
                    .uri("/chat")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(body)
                    .retrieve()
                    .bodyToMono(Map.class)
                    .block();

            if (res == null) return "(keine Antwort)";

            Object reply = res.get("reply");
            if (reply != null) return reply.toString();

            return "(keine Antwort)";

        } catch (WebClientResponseException e) {
            return "⚠️ Fehler: HTTP " + e.getStatusCode().value();
        } catch (Exception e) {
            return "⚠️ Unerwarteter Fehler: " + e.getMessage();
        }
}
}