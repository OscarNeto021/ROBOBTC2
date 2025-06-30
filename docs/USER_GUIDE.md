# Guia do Usuário: BTC-PERP Absolute Trader

## Manual Completo de Uso

**Versão:** 1.0  
**Data:** Janeiro 2025  

---

## 📋 Índice

1. [Introdução](#introdução)
2. [Instalação Rápida](#instalação-rápida)
3. [Primeiros Passos](#primeiros-passos)
4. [Interface Web](#interface-web)
5. [Configurações](#configurações)
6. [Operação do Sistema](#operação-do-sistema)
7. [Monitoramento](#monitoramento)
8. [Solução de Problemas](#solução-de-problemas)
9. [FAQ](#faq)
10. [Suporte](#suporte)

---

## 🎯 Introdução

O **BTC-PERP Absolute Trader** é um sistema de trading algorítmico que utiliza Machine Learning para operar contratos futuros perpétuos de Bitcoin. Este guia irá ajudá-lo a configurar, usar e monitorar o sistema.

### ⚠️ Aviso Importante

**Este sistema é fornecido apenas para fins educacionais e de pesquisa. Trading de criptomoedas envolve riscos significativos. Use por sua própria conta e risco.**

### 🎯 O que o Sistema Faz

- **Coleta dados** de mercado em tempo real
- **Analisa padrões** usando Machine Learning
- **Gera sinais** de compra e venda
- **Executa ordens** automaticamente (modo simulado)
- **Monitora riscos** continuamente
- **Fornece métricas** de performance

---

## 🚀 Instalação Rápida

### Opção 1: Docker (Recomendado)

```bash
# 1. Clonar repositório
git clone <repository-url>
cd btc_perp_trader

# 2. Iniciar com Docker
docker-compose up -d

# 3. Acessar interface
# Abrir http://localhost:5000 no navegador
```

### Opção 2: Instalação Local

```bash
# 1. Pré-requisitos
# Python 3.11+, pip, git

# 2. Clonar e configurar
git clone <repository-url>
cd btc_perp_trader
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Configurar
cp config/.env.example config/.env
# Editar config/.env se necessário

# 5. Iniciar API
cd trading_api
python src/main.py

# 6. Acessar interface
# Abrir http://localhost:5000 no navegador
```

### Verificação da Instalação

1. Acesse `http://localhost:5000`
2. Você deve ver o dashboard do sistema
3. Status deve mostrar "Não Inicializado"

---

## 🏁 Primeiros Passos

### Passo 1: Inicializar o Sistema

1. **Acesse a interface web** em `http://localhost:5000`
2. **Clique em "Inicializar Sistema"**
3. **Aguarde** a mensagem de confirmação
4. **Verifique** se o status mudou para "Parado"

### Passo 2: Configurar Parâmetros (Opcional)

Edite o arquivo `config/config.yaml`:

```yaml
trading:
  initial_capital: 100000  # Capital inicial em USD
  commission_rate: 0.001   # Taxa de comissão (0.1%)
  max_position_size: 0.1   # Máximo 10% por posição

risk_management:
  max_daily_loss: 0.05     # Máximo 5% de perda diária
  max_drawdown: 0.15       # Máximo 15% de drawdown
```

### Passo 3: Iniciar o Trading

1. **Clique em "Iniciar Trading"**
2. **Aguarde** a confirmação
3. **Observe** o status mudar para "Rodando"
4. **Monitore** as métricas no dashboard

### Passo 4: Atualizar Dados de Mercado

Para simular dados de mercado:

1. **Digite um preço** no campo "Atualizar Preço BTC-USDT"
2. **Clique em "Atualizar Preço"**
3. **Observe** as mudanças no sistema

---

## 🌐 Interface Web

### Dashboard Principal

A interface web é dividida em seções:

#### 📊 Status do Sistema
- **Status**: Estado atual (Não Inicializado/Parado/Rodando)
- **Saldo**: Capital disponível
- **Equity Total**: Valor total da conta
- **Última Atualização**: Timestamp da última atualização

#### 📈 Performance
- **Total de Trades**: Número de operações realizadas
- **Ordens Abertas**: Ordens pendentes
- **Posições**: Posições abertas
- **Win Rate**: Taxa de acerto (quando disponível)

#### ⚡ Ações Rápidas
- **Atualizar Preço**: Simular mudança de preço
- **Controles**: Inicializar, Iniciar, Parar sistema

#### 📋 Colocar Ordem Manual
Formulário para ordens manuais:
- **Símbolo**: Par de trading (BTC-USDT)
- **Lado**: Comprar ou Vender
- **Tipo**: Market ou Limit
- **Quantidade**: Quantidade a negociar
- **Preço**: Preço para ordens limit

#### 📊 Tabelas
- **Posições Atuais**: Posições abertas com PnL
- **Ordens Abertas**: Ordens pendentes com opção de cancelar

### Navegação

- **Atualização Automática**: Dados atualizados a cada 5 segundos
- **Responsivo**: Funciona em desktop e mobile
- **Tempo Real**: Métricas em tempo real

---

## ⚙️ Configurações

### Arquivo Principal: `config/config.yaml`

#### Seção Trading
```yaml
trading:
  initial_capital: 100000      # Capital inicial
  commission_rate: 0.001       # Taxa de comissão
  max_position_size: 0.1       # Tamanho máximo da posição
  risk_free_rate: 0.02         # Taxa livre de risco
```

#### Seção Data
```yaml
data:
  symbols: ["BTCUSDT"]         # Símbolos a negociar
  timeframes: ["5m", "15m"]    # Timeframes dos dados
  lookback_days: 30            # Dias de histórico
```

#### Seção Models
```yaml
models:
  xgboost:
    n_estimators: 1000         # Número de árvores
    max_depth: 6               # Profundidade máxima
    learning_rate: 0.1         # Taxa de aprendizado
  
  lstm:
    sequence_length: 60        # Comprimento da sequência
    hidden_size: 128           # Tamanho da camada oculta
    epochs: 100                # Épocas de treinamento
```

#### Seção Risk Management
```yaml
risk_management:
  max_position_size: 0.1       # Máximo por posição
  max_portfolio_risk: 0.2      # Risco máximo do portfólio
  max_daily_loss: 0.05         # Perda diária máxima
  max_drawdown: 0.15           # Drawdown máximo
  stop_loss_pct: 0.02          # Stop loss padrão
  take_profit_pct: 0.04        # Take profit padrão
```

### Variáveis de Ambiente: `config/.env`

```bash
# Configurações de API (opcional)
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Configurações de logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading.log

# Configurações de banco de dados
DATABASE_URL=sqlite:///data/trading.db
```

### Aplicando Configurações

1. **Edite** os arquivos de configuração
2. **Reinicie** o sistema
3. **Reinicialize** através da interface web

---

## 🎮 Operação do Sistema

### Estados do Sistema

#### 1. Não Inicializado
- **Descrição**: Sistema não configurado
- **Ações Disponíveis**: Inicializar
- **Indicador**: Círculo laranja

#### 2. Parado
- **Descrição**: Sistema configurado mas não operando
- **Ações Disponíveis**: Iniciar Trading, Configurar
- **Indicador**: Círculo vermelho

#### 3. Rodando
- **Descrição**: Sistema operando automaticamente
- **Ações Disponíveis**: Parar Trading, Monitorar
- **Indicador**: Círculo verde

### Fluxo de Operação

```
Inicializar → Configurar → Iniciar → Monitorar → Parar
     ↑                                            ↓
     ←←←←←←←←←← Reinicializar ←←←←←←←←←←←←←←←←←←←←←←
```

### Operações Manuais

#### Colocar Ordem
1. **Preencha** o formulário de ordem
2. **Selecione** tipo (Market/Limit)
3. **Clique** em "Colocar Ordem"
4. **Aguarde** confirmação

#### Cancelar Ordem
1. **Localize** a ordem na tabela
2. **Clique** em "Cancelar"
3. **Aguarde** confirmação

#### Atualizar Preço
1. **Digite** novo preço
2. **Clique** em "Atualizar Preço"
3. **Observe** reação do sistema

### Automação

Quando em modo "Rodando":
- **Coleta dados** automaticamente
- **Gera sinais** baseado nos modelos
- **Executa ordens** conforme estratégia
- **Monitora riscos** continuamente
- **Ajusta posições** quando necessário

---

## 📊 Monitoramento

### Métricas Principais

#### Performance
- **Total Return**: Retorno total desde o início
- **Sharpe Ratio**: Retorno ajustado ao risco
- **Win Rate**: Percentual de trades vencedores
- **Profit Factor**: Razão lucro/prejuízo

#### Risco
- **Maximum Drawdown**: Maior perda peak-to-trough
- **Current Drawdown**: Drawdown atual
- **VaR**: Value at Risk (perda máxima esperada)
- **Position Size**: Tamanho atual das posições

#### Operacional
- **Total Trades**: Número de operações
- **Open Orders**: Ordens pendentes
- **Active Positions**: Posições abertas
- **Last Update**: Última atualização

### Alertas e Notificações

O sistema monitora automaticamente:

#### Alertas de Risco
- **Drawdown Excessivo**: > 15%
- **Perda Diária**: > 5%
- **Posição Grande**: > 10%
- **Volatilidade Alta**: > 50%

#### Alertas Operacionais
- **Falha de Dados**: Sem dados por > 5 min
- **Erro de Modelo**: Falha na previsão
- **Ordem Rejeitada**: Ordem não executada
- **Sistema Parado**: Parada inesperada

### Logs do Sistema

#### Localização
- **Arquivo**: `logs/trading.log`
- **Formato**: Timestamp, Level, Message
- **Rotação**: Diária, mantém 30 dias

#### Níveis de Log
- **DEBUG**: Informações detalhadas
- **INFO**: Operações normais
- **WARNING**: Situações de atenção
- **ERROR**: Erros que requerem ação
- **CRITICAL**: Falhas críticas do sistema

#### Exemplo de Log
```
2024-01-15 10:30:15 INFO Trading bot started
2024-01-15 10:30:16 INFO Market data updated: BTC-USDT @ 50000.00
2024-01-15 10:30:17 INFO Signal generated: BUY (confidence: 0.75)
2024-01-15 10:30:18 INFO Order placed: BUY 0.1 BTC @ MARKET
2024-01-15 10:30:19 INFO Order filled: BUY 0.1 BTC @ 50005.00
```

---

## 🔧 Solução de Problemas

### Problemas Comuns

#### 1. Sistema Não Inicializa

**Sintomas:**
- Erro ao clicar "Inicializar Sistema"
- Mensagem de erro na interface

**Soluções:**
```bash
# Verificar logs
tail -f logs/trading.log

# Verificar dependências
pip install -r requirements.txt

# Verificar configuração
cat config/config.yaml

# Reiniciar sistema
docker-compose restart  # ou
python trading_api/src/main.py
```

#### 2. Interface Web Não Carrega

**Sintomas:**
- Página não abre em localhost:5000
- Erro de conexão

**Soluções:**
```bash
# Verificar se servidor está rodando
ps aux | grep python

# Verificar porta
netstat -tlnp | grep 5000

# Verificar logs do Flask
tail -f logs/trading.log

# Reiniciar servidor
pkill -f "python.*main.py"
python trading_api/src/main.py
```

#### 3. Dados Não Atualizam

**Sintomas:**
- Métricas não mudam
- "Última Atualização" parada

**Soluções:**
```bash
# Verificar conectividade
ping api.binance.com

# Verificar logs de dados
grep "data" logs/trading.log

# Forçar atualização
# Usar botão "Atualizar Dados" na interface
```

#### 4. Ordens Não Executam

**Sintomas:**
- Ordens ficam pendentes
- Erro ao colocar ordem

**Soluções:**
```bash
# Verificar saldo
# Saldo deve ser suficiente para a ordem

# Verificar configurações
grep "commission" config/config.yaml

# Verificar logs de execução
grep "order" logs/trading.log
```

#### 5. Performance Baixa

**Sintomas:**
- Sistema lento
- Alto uso de CPU/memória

**Soluções:**
```bash
# Verificar recursos
top
free -h

# Otimizar configurações
# Reduzir frequência de dados
# Simplificar modelos

# Verificar logs de performance
grep "performance" logs/trading.log
```

### Comandos de Diagnóstico

#### Verificar Status Geral
```bash
# Status dos serviços
docker-compose ps

# Logs recentes
docker-compose logs --tail=50

# Uso de recursos
docker stats
```

#### Verificar Configuração
```bash
# Validar YAML
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Verificar variáveis de ambiente
cat config/.env

# Testar importações
python -c "from src.execution.execution_engine import ExecutionEngine"
```

#### Verificar Dados
```bash
# Verificar arquivos de dados
ls -la data/

# Verificar logs de dados
grep "data" logs/trading.log | tail -20

# Testar conectividade
curl -s "https://api.binance.com/api/v3/ping"
```

### Recuperação de Falhas

#### Falha Total do Sistema
1. **Parar** todos os processos
2. **Verificar** logs para identificar causa
3. **Corrigir** problema identificado
4. **Reiniciar** sistema
5. **Verificar** funcionamento

#### Corrupção de Dados
1. **Parar** sistema
2. **Backup** dados corrompidos
3. **Restaurar** de backup ou recriar
4. **Reiniciar** sistema
5. **Validar** integridade

#### Falha de Modelo
1. **Parar** trading
2. **Verificar** logs de modelo
3. **Retreinar** modelo se necessário
4. **Testar** modelo
5. **Reiniciar** trading

---

## ❓ FAQ

### Perguntas Gerais

**Q: O sistema opera com dinheiro real?**
A: Não, o sistema atual é apenas simulado. Não há integração com exchanges reais.

**Q: Posso usar com outras criptomoedas?**
A: Atualmente suporta apenas BTC-USDT, mas pode ser estendido para outros pares.

**Q: Preciso de conhecimento técnico?**
A: Conhecimento básico é recomendado, mas a interface web simplifica o uso.

**Q: O sistema funciona 24/7?**
A: Sim, quando configurado corretamente, opera continuamente.

### Perguntas Técnicas

**Q: Como treinar novos modelos?**
A: Use o módulo de backtesting ou execute scripts de treinamento diretamente.

**Q: Posso modificar as estratégias?**
A: Sim, o código é modular e permite customizações.

**Q: Como adicionar novos indicadores?**
A: Edite o arquivo `src/features/feature_engineering.py`.

**Q: Posso usar meus próprios dados?**
A: Sim, substitua os coletores de dados pelos seus.

### Perguntas de Performance

**Q: Qual o hardware mínimo?**
A: 4GB RAM, 2 CPU cores, 10GB disco.

**Q: Como melhorar a performance?**
A: Use SSD, mais RAM, otimize configurações.

**Q: O sistema consome muitos recursos?**
A: Uso moderado, ~1-2GB RAM, 10-20% CPU.

### Perguntas de Segurança

**Q: É seguro usar?**
A: Para fins educacionais sim, mas não use com dinheiro real sem testes extensivos.

**Q: Como proteger as configurações?**
A: Use variáveis de ambiente para dados sensíveis.

**Q: Há logs de auditoria?**
A: Sim, todos os trades e decisões são logados.

---

## 📞 Suporte

### Documentação

- **README.md**: Visão geral do projeto
- **TECHNICAL_REPORT.md**: Documentação técnica detalhada
- **USER_GUIDE.md**: Este guia do usuário
- **API Documentation**: Disponível em `/api/docs`

### Comunidade

- **GitHub Issues**: Para reportar bugs
- **GitHub Discussions**: Para discussões gerais
- **Wiki**: Tutoriais e exemplos adicionais

### Recursos de Aprendizado

#### Livros Recomendados
- "Advances in Financial Machine Learning" - López de Prado
- "Algorithmic Trading" - Ernest Chan
- "Python for Finance" - Yves Hilpisch

#### Cursos Online
- Machine Learning for Trading (Coursera)
- Algorithmic Trading (Udemy)
- Python for Finance (edX)

#### Comunidades
- QuantConnect Community
- Quantitative Finance Stack Exchange
- Reddit r/algotrading

### Contato

Para suporte técnico:
1. **Verifique** este guia primeiro
2. **Consulte** os logs do sistema
3. **Procure** em issues existentes no GitHub
4. **Abra** nova issue se necessário

### Contribuição

Interessado em contribuir?
1. **Fork** o repositório
2. **Implemente** melhorias
3. **Teste** suas mudanças
4. **Abra** Pull Request

---

## 📝 Notas Finais

### Disclaimer Legal

**AVISO IMPORTANTE**: Este software é fornecido "como está" para fins educacionais e de pesquisa. Os desenvolvedores não se responsabilizam por perdas financeiras. Trading envolve riscos significativos.

### Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

### Atualizações

Este guia é atualizado regularmente. Verifique a versão mais recente no repositório.

---

**Desenvolvido com ❤️ para a comunidade de trading algorítmico**

*Última atualização: Janeiro 2025*
*Versão do Guia: 1.0*

