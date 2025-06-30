import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Button } from '@/components/ui/button.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, Target, Activity, BarChart3, PieChart as PieChartIcon, Settings } from 'lucide-react'
import './App.css'

// Dados simulados baseados no relatório HTML
const tradingData = {
  overview: {
    startValue: 100000,
    endValue: 131751.60,
    totalReturn: 31.75,
    benchmarkReturn: 18.74,
    maxDrawdown: 11.97,
    sharpeRatio: 13.90,
    totalTrades: 10,
    winRate: 40.0,
    profitFactor: 4.00,
    expectancy: 3175.16
  },
  trades: {
    totalTrades: 10,
    winningTrades: 4,
    losingTrades: 6,
    bestTrade: 30.44,
    worstTrade: -2.89,
    avgWinningTrade: 10.57,
    avgLosingTrade: -1.74
  },
  performance: [
    { date: '2023-01-01', value: 100000, benchmark: 100000 },
    { date: '2023-01-02', value: 105000, benchmark: 102000 },
    { date: '2023-01-03', value: 108000, benchmark: 104000 },
    { date: '2023-01-04', value: 112000, benchmark: 106000 },
    { date: '2023-01-05', value: 118000, benchmark: 108000 },
    { date: '2023-01-06', value: 125000, benchmark: 112000 },
    { date: '2023-01-07', value: 131751, benchmark: 118742 }
  ],
  monthlyReturns: [
    { month: 'Jan', return: 8.5 },
    { month: 'Feb', return: 12.3 },
    { month: 'Mar', return: 6.8 },
    { month: 'Apr', return: 15.2 },
    { month: 'May', return: -3.1 },
    { month: 'Jun', return: 9.7 }
  ],
  tradeDistribution: [
    { name: 'Trades Ganhos', value: 4, color: '#10b981' },
    { name: 'Trades Perdidos', value: 6, color: '#ef4444' }
  ]
}

function App() {
  const [selectedTab, setSelectedTab] = useState('overview')
  const [isLive, setIsLive] = useState(false)

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'USD'
    }).format(value)
  }

  const formatPercentage = (value) => {
    return `${value.toFixed(2)}%`
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <BarChart3 className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
                Robô BTC Trading
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <Badge variant={isLive ? "default" : "secondary"}>
                {isLive ? "AO VIVO" : "BACKTEST"}
              </Badge>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Configurações
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Visão Geral</TabsTrigger>
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="trades">Trades</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* Key Metrics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Valor Total</CardTitle>
                  <DollarSign className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatCurrency(tradingData.overview.endValue)}</div>
                  <p className="text-xs text-muted-foreground">
                    +{formatCurrency(tradingData.overview.endValue - tradingData.overview.startValue)} desde o início
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Retorno Total</CardTitle>
                  <TrendingUp className="h-4 w-4 text-green-600" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-green-600">
                    +{formatPercentage(tradingData.overview.totalReturn)}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    vs {formatPercentage(tradingData.overview.benchmarkReturn)} benchmark
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Taxa de Acerto</CardTitle>
                  <Target className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{formatPercentage(tradingData.overview.winRate)}</div>
                  <Progress value={tradingData.overview.winRate} className="mt-2" />
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                  <Activity className="h-4 w-4 text-muted-foreground" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{tradingData.overview.sharpeRatio.toFixed(2)}</div>
                  <p className="text-xs text-muted-foreground">
                    Excelente performance ajustada ao risco
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Additional Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Métricas de Risco</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Max Drawdown</span>
                    <span className="text-sm font-medium text-red-600">
                      -{formatPercentage(tradingData.overview.maxDrawdown)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Profit Factor</span>
                    <span className="text-sm font-medium text-green-600">
                      {tradingData.overview.profitFactor.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Expectancy</span>
                    <span className="text-sm font-medium">
                      {formatCurrency(tradingData.overview.expectancy)}
                    </span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Estatísticas de Trades</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Total de Trades</span>
                    <span className="text-sm font-medium">{tradingData.trades.totalTrades}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Trades Ganhos</span>
                    <span className="text-sm font-medium text-green-600">{tradingData.trades.winningTrades}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Trades Perdidos</span>
                    <span className="text-sm font-medium text-red-600">{tradingData.trades.losingTrades}</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Performance de Trades</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Melhor Trade</span>
                    <span className="text-sm font-medium text-green-600">
                      +{formatPercentage(tradingData.trades.bestTrade)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Pior Trade</span>
                    <span className="text-sm font-medium text-red-600">
                      {formatPercentage(tradingData.trades.worstTrade)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Média Ganhos</span>
                    <span className="text-sm font-medium text-green-600">
                      +{formatPercentage(tradingData.trades.avgWinningTrade)}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Performance Tab */}
          <TabsContent value="performance" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Curva de Equity</CardTitle>
                <CardDescription>
                  Comparação da performance do robô vs benchmark
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={tradingData.performance}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip formatter={(value) => formatCurrency(value)} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#2563eb" 
                      strokeWidth={2}
                      name="Robô BTC"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="benchmark" 
                      stroke="#64748b" 
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      name="Benchmark"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Retornos Mensais</CardTitle>
                <CardDescription>
                  Performance mensal do robô de trading
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={tradingData.monthlyReturns}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="return" fill="#2563eb" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Trades Tab */}
          <TabsContent value="trades" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Distribuição de Trades</CardTitle>
                  <CardDescription>
                    Proporção entre trades ganhos e perdidos
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={tradingData.tradeDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, value }) => `${name}: ${value}`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {tradingData.tradeDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Resumo de Performance</CardTitle>
                  <CardDescription>
                    Principais métricas de trading
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Taxa de Acerto</span>
                      <span className="font-medium">{formatPercentage(tradingData.overview.winRate)}</span>
                    </div>
                    <Progress value={tradingData.overview.winRate} className="h-2" />
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Profit Factor</span>
                      <span className="font-medium text-green-600">{tradingData.overview.profitFactor.toFixed(2)}</span>
                    </div>
                    <Progress value={Math.min(tradingData.overview.profitFactor * 25, 100)} className="h-2" />
                  </div>

                  <div className="pt-4 border-t">
                    <div className="grid grid-cols-2 gap-4 text-center">
                      <div>
                        <div className="text-2xl font-bold text-green-600">{tradingData.trades.winningTrades}</div>
                        <div className="text-sm text-muted-foreground">Trades Ganhos</div>
                      </div>
                      <div>
                        <div className="text-2xl font-bold text-red-600">{tradingData.trades.losingTrades}</div>
                        <div className="text-sm text-muted-foreground">Trades Perdidos</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Análise de Risco</CardTitle>
                  <CardDescription>
                    Métricas avançadas de risco e retorno
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">{tradingData.overview.sharpeRatio.toFixed(2)}</div>
                      <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
                    </div>
                    <div className="text-center p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
                      <div className="text-2xl font-bold text-red-600">{formatPercentage(tradingData.overview.maxDrawdown)}</div>
                      <div className="text-sm text-muted-foreground">Max Drawdown</div>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex justify-between">
                      <span className="text-sm">Volatilidade Anualizada</span>
                      <span className="text-sm font-medium">12.5%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Beta vs BTC</span>
                      <span className="text-sm font-medium">0.85</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Alpha Anualizado</span>
                      <span className="text-sm font-medium text-green-600">8.2%</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Machine Learning</CardTitle>
                  <CardDescription>
                    Status do modelo e aprendizado
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Modelo Ativo</span>
                    <Badge variant="default">River Online ML</Badge>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Acurácia do Modelo</span>
                      <span className="font-medium">73.2%</span>
                    </div>
                    <Progress value={73.2} className="h-2" />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Confiança da Predição</span>
                      <span className="font-medium">85.7%</span>
                    </div>
                    <Progress value={85.7} className="h-2" />
                  </div>

                  <div className="pt-4 border-t space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Samples Processados</span>
                      <span className="font-medium">1,247</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Última Atualização</span>
                      <span className="font-medium">2 min atrás</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Configurações do Sistema</CardTitle>
                <CardDescription>
                  Parâmetros e configurações do robô
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="space-y-2">
                    <h4 className="font-medium">Trading</h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Capital Inicial</span>
                        <span>{formatCurrency(tradingData.overview.startValue)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Risk per Trade</span>
                        <span>2%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Stop Loss</span>
                        <span>3%</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-medium">Modelo ML</h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Algoritmo</span>
                        <span>XGBoost + River</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Features</span>
                        <span>42</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Timeframe</span>
                        <span>5min</span>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-medium">Dados</h4>
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Exchange</span>
                        <span>Binance</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Par</span>
                        <span>BTC/USDT</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">News API</span>
                        <span>Ativo</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}

export default App

