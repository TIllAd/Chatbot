package de.fau.wisochatbot;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class ChatApiController {

  @GetMapping("/api/echo")
  public String echo(@RequestParam String msg) {
    return "Du: " + msg + "\nBot: (hier kommt später RAG/LLM rein)";
  }
}